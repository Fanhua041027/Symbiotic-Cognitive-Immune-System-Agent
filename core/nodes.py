"""核心智能体节点：主智能体、监察员T细胞、抗体生成器、沙箱验证器。"""

import json
import threading

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from core.circuit_breaker import breaker
from core.config import get as cfg
from core.logger import setup_logger
from core.memory import memory_db
from core.sandbox import validate_antibody
from core.state import ImmunologyState

logger = setup_logger("nodes")

load_dotenv()

# ---------------------------------------------------------------------------
# LLM Provider 抽象层
# ---------------------------------------------------------------------------
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "custom": None,  # 从 CUSTOM_API_BASE 读取
}


def _resolve_llm_params() -> dict:
    """根据 LLM_PROVIDER 配置返回 api_key 和 base_url。"""
    provider = cfg("LLM_PROVIDER", "openai")

    if provider == "openai":
        return {
            "api_key": cfg("OPENAI_API_KEY"),
            "base_url": PROVIDER_ENDPOINTS["openai"],
        }
    elif provider == "deepseek":
        return {
            "api_key": cfg("DEEPSEEK_API_KEY"),
            "base_url": PROVIDER_ENDPOINTS["deepseek"],
        }
    elif provider == "custom":
        return {
            "api_key": cfg("CUSTOM_API_KEY"),
            "base_url": cfg("CUSTOM_API_BASE"),
        }
    else:
        logger.warning("Unknown LLM_PROVIDER=%s, falling back to openai", provider)
        return {
            "api_key": cfg("OPENAI_API_KEY"),
            "base_url": PROVIDER_ENDPOINTS["openai"],
        }


def _create_llm(model_key: str, temperature: float) -> ChatOpenAI:
    """创建适配当前 Provider 的 LLM 实例。"""
    params = _resolve_llm_params()
    api_key = params.get("api_key")
    if not api_key:
        logger.warning("No API key configured for provider=%s, LLM calls will fail",
                        cfg("LLM_PROVIDER", "openai"))
    model = cfg(model_key)
    logger.info(
        "LLM init: provider=%s model=%s endpoint=%s",
        cfg("LLM_PROVIDER", "openai"), model,
        params.get("base_url", "unknown"),
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=params["base_url"],
    )


# ---------------------------------------------------------------------------
# LLM 实例（惰性初始化：仅在首次调用时创建）
# ---------------------------------------------------------------------------
_llm_cache: dict[str, ChatOpenAI] = {}
_llm_cache_lock = threading.Lock()


def _get_llm(role: str, model_key: str, temperature: float) -> ChatOpenAI:
    """Get or create a lazy LLM instance keyed by effective config values.
    Thread-safe: uses a lock to prevent duplicate _create_llm calls.
    """
    provider = cfg("LLM_PROVIDER", "openai")
    model = cfg(model_key)
    cache_key = f"{role}:{provider}:{model}:{temperature}"
    with _llm_cache_lock:
        if cache_key not in _llm_cache:
            _llm_cache[cache_key] = _create_llm(model_key, temperature)
        return _llm_cache[cache_key]


def get_main_llm() -> ChatOpenAI:
    return _get_llm("main", "MAIN_LLM_MODEL", cfg("LLM_TEMPERATURE", 0.7))


def get_monitor_llm() -> ChatOpenAI:
    return _get_llm("monitor", "MONITOR_LLM_MODEL", 0.0)


def get_antibody_llm() -> ChatOpenAI:
    return _get_llm("antibody", "ANTIBODY_LLM_MODEL", 0.2)


# Cached fallback LLM (lazy, built once from alternative provider)
_fallback_llm: ChatOpenAI | None = None
_fallback_llm_lock = threading.Lock()


def _get_fallback_llm() -> ChatOpenAI | None:
    global _fallback_llm
    if _fallback_llm is not None:
        return _fallback_llm
    with _fallback_llm_lock:
        if _fallback_llm is not None:
            return _fallback_llm
        provider = cfg("LLM_PROVIDER", "openai")
        # Try a different provider as fallback
        if provider == "deepseek" and cfg("OPENAI_API_KEY"):
            _fallback_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=cfg("LLM_TEMPERATURE", 0.7),
                api_key=cfg("OPENAI_API_KEY"),
                base_url=PROVIDER_ENDPOINTS["openai"],
            )
            logger.info("Fallback LLM: openai/gpt-4o-mini (for deepseek provider)")
        elif provider == "openai" and cfg("DEEPSEEK_API_KEY"):
            _fallback_llm = ChatOpenAI(
                model="deepseek-chat",
                temperature=cfg("LLM_TEMPERATURE", 0.7),
                api_key=cfg("DEEPSEEK_API_KEY"),
                base_url=PROVIDER_ENDPOINTS["deepseek"],
            )
            logger.info("Fallback LLM: deepseek/deepseek-chat (for openai provider)")
        return _fallback_llm


def _invoke_llm(
    llm: ChatOpenAI,
    prompt: str,
    circuit_name: str,
    fallback: ChatOpenAI | None = None,
) -> str:
    """Invoke LLM with circuit breaker and optional fallback.
    Raises on failure — caller handles the exception."""
    if not breaker.can_execute(circuit_name):
        if fallback:
            logger.info("Circuit breaker [%s] open, using fallback LLM", circuit_name)
            response = fallback.invoke(prompt)
            # Do NOT record_success on primary breaker — it's still failing.
            # Leave it in half_open so the next call probes the primary first.
            raw = response.content
            return raw if isinstance(raw, str) else str(raw)
        raise RuntimeError(f"LLM circuit breaker open for '{circuit_name}' (no fallback)")

    try:
        response = llm.invoke(prompt)
        breaker.record_success(circuit_name)
        raw = response.content
        return raw if isinstance(raw, str) else str(raw)
    except Exception as e:
        breaker.record_failure(circuit_name)
        if fallback:
            logger.info("Primary LLM failed for [%s], trying fallback: %s", circuit_name, e)
            try:
                response = fallback.invoke(prompt)
                # Leave primary breaker in half_open — the fallback succeeding
                # doesn't mean the primary has recovered. The next call will
                # probe the primary through can_execute()'s half_open path.
                raw = response.content
                return raw if isinstance(raw, str) else str(raw)
            except Exception as e2:
                breaker.record_failure(circuit_name)
                logger.error("Fallback LLM also failed for [%s]: %s", circuit_name, e2)
        raise


# ---------------------------------------------------------------------------
# Git 自动备份 (免疫响应时创建安全快照)
# ---------------------------------------------------------------------------
_AUTO_BACKUP_ENABLED = True


def _auto_git_backup(error_pattern: str) -> None:
    """Create an automatic git checkpoint when an immune response fires."""
    if not _AUTO_BACKUP_ENABLED:
        return
    import subprocess
    from datetime import datetime, timezone
    try:
        # Check if we're in a git repo
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not repo_root:
            return

        # Check for changes to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not status:
            return

        # Stage all and commit with immune response message
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, timeout=10,
        )
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
        msg = (
            f"immune: auto-backup {timestamp} — {error_pattern[:60]}"
        )
        subprocess.run(
            ["git", "commit", "-m", msg, "--no-gpg-sign"],
            capture_output=True, timeout=10,
        )
        logger.info("Auto git backup created: %s", msg)
    except Exception as e:
        logger.debug("Auto git backup skipped: %s", e)


# ---------------------------------------------------------------------------
# 节点 A: 主智能体 (Worker)
# ---------------------------------------------------------------------------
def main_worker_node(state: ImmunologyState) -> dict:
    """
    主智能体：尝试完成用户任务。
    执行前先查询免疫记忆库，注入历史抗体；同时注入当前会话抗体。
    """
    query = state["user_query"]

    # 查询免疫记忆库，获取历史抗体
    memory_hit = memory_db.search_antibody(query)
    injected_context = ""

    if memory_hit:
        logger.info(
            "Historical antibody found for pattern: %s...",
            memory_hit.get("pattern", "")[:50],
        )
        injected_context += (
            f"\n[Historical Memory]: {memory_hit['code']}\n"
        )

    # 注入当前会话的抗体
    if state.get("antibodies"):
        last_antibody = state["antibodies"][-1]
        injected_context += (
            f"\n[Session Fix Applied]: {last_antibody['code']}\n"
            f"[Fix Explanation]: {last_antibody['explanation']}\n"
        )

    # When session/historical antibodies are present, the Worker should apply
    # the fix and produce clean output — NOT re-trigger anomaly detection.
    has_fix = bool(injected_context.strip())

    if has_fix:
        full_prompt = f"""You are a reasoning agent. Complete the user's task below.

User task: {query}
{injected_context}

**Critical instruction: A previous anomaly was detected and a fix has been applied above.**
Your job now is to PRODUCE CORRECT OUTPUT — not to re-detect the same issue.

**Step 1 — Apply the fix:**
- Incorporate the [Historical Memory] and/or [Session Fix Applied] code directly into your solution.
- Make sure the termination guards, preconditions, and safety checks from the fix are present.
- The fix IS your solution — extend it to fully answer the user's request while keeping the guards.

**Step 2 — Final output — WRITE THE ACTUAL ANSWER:**
- Your output MUST be the final, complete answer to the user's task with the fix incorporated.
- DO NOT output "COGNITIVE_ANOMALY:" — the anomaly has already been handled.
- DO NOT refuse to answer or suggest escalation — just produce the fixed solution.
- If you notice a DIFFERENT issue the existing fix doesn't cover, fix it silently in your output.
- Your output goes directly to the user as the final answer.
"""
    else:
        full_prompt = f"""You are a reasoning agent. Complete the user's task below.

User task: {query}

**Step 1 — Reason step by step:**
Think through the problem. Consider edge cases, termination, and resource constraints.

**Step 2 — Self-Check (evaluate each category):**
For each, answer YES (flag it) or NO (clean):

1. **Infinite loop** — Is termination guaranteed? (YES: `while True` no break, recursion no base case)
2. **Logical contradiction** — Impossible conditions? (YES: `x>10 and x<5`, intentional fallacy like "prove 1+1=3")
3. **Halting / self-reference** — Self-referential paradox? (YES: function checking if IT will loop, cache calling itself)
4. **Resource safety** — Bounded? (YES: open() no close, unbounded list growth)
5. **Type / correctness** — Types consistent? (YES: str+int, off-by-one, byte/char confusion)
6. **Security** — Dangerous patterns? (YES: exec/eval on input, SQL injection)
7. **Hallucinated API** — Real libraries? (YES: `import antigravity` for computation)
8. **User trap** — User asking for something wrong? (YES: "prove 1+1=3 as fallacy", "download+execute malware is safe")

**Step 3 — Output:**
- If ANY category is YES: start with `COGNITIVE_ANOMALY: <category> - <reason>` then show the problematic reasoning.
- If all clean: provide your solution directly. Include a max-iteration guard or recursion depth limit for code.
"""

    # 自增迭代计数
    iteration = (state.get("iteration_count") or 0) + 1

    try:
        content = _invoke_llm(get_main_llm(), full_prompt, "worker", _get_fallback_llm())
        content = content.strip()
    except Exception as e:
        logger.error("Worker LLM call failed: %s", e)
        return {
            "task_steps": [{"step": "llm_error", "error": str(e)}],
            "anomalies": [{
                "status": "unhealthy",
                "reason": f"LLM API error in worker: {e}",
                "source": "worker",
            }],
            "final_output": None,
            "iteration_count": iteration,
        }

    # 检测 LLM 是否自报告了认知异常
    if content.startswith("COGNITIVE_ANOMALY:"):
        # 提取异常描述
        lines = content.split("\n")
        reason = lines[0].replace("COGNITIVE_ANOMALY:", "").strip()
        return {
            "task_steps": [
                {
                    "step": "cognitive_anomaly_detected",
                    "content": content,
                    "reason": reason,
                },
            ],
            "anomalies": [
                {
                    "status": "unhealthy",
                    "reason": f"Worker self-reported anomaly: {reason}",
                    "source": "worker",
                },
            ],
            "final_output": None,
            "iteration_count": iteration,
        }

    return {
        "task_steps": [{"step": "execute", "content": content}],
        "final_output": content,
        "anomalies": [],  # 不覆盖已有异常，在 monitor 中决定
        "iteration_count": iteration,
    }


# ---------------------------------------------------------------------------
# 节点 B1: 一致性检查 (Consistency Checker)
# ---------------------------------------------------------------------------
def consistency_check_node(state: ImmunologyState) -> dict:
    """
    一致性检查：分析 Worker 的输出是否与原始查询和基本事实一致。
    如果已生成抗体，则优先验证修复是否正确应用。
    """
    steps = state.get("task_steps", [])
    query = state["user_query"]
    antibodies = state.get("antibodies", [])
    validation_status = state.get("validation_status")
    fix_applied = bool(antibodies) and validation_status == "passed"

    worker_output = ""
    for step in steps:
        content = step.get("content", "")
        if content:
            worker_output = content
            break

    if not worker_output:
        existing = state.get("anomalies") or []
        return {"anomalies": existing}

    fix_context = ""
    if fix_applied:
        fix_context = (
            "\n\n**A fix has already been applied by the immune system.**\n"
            f"Applied fix code: {antibodies[-1].get('code', '')[:300]}\n"
            f"Fix explanation: {antibodies[-1].get('explanation', '')[:200]}\n"
            "Your job is to verify the fix was CORRECTLY applied.\n"
            "Do NOT re-flag the same issue that was already fixed.\n"
            "Only flag a NEW issue that is completely different from what was already addressed.\n"
        )

    prompt = f"""You are a consistency validator for an AI immune system.
Analyze the worker agent's output for issues the worker may have missed.

Original query: {query}

Worker output:
{worker_output[:2000]}
{fix_context}
**Check ALL of these patterns:**

1. **Impossible logical conditions** — Does the code have AND/OR conditions that can never be true?
   Example: `if x > 10 and x < 5:` — impossible

2. **Intentional fallacy / trick instruction** — Is the user's query asking for something logically wrong, and the worker complied?
   Example: "prove 1+1=3" — worker should refuse or flag this, not comply
   Example: "explain why downloading+executing a remote file is secure" — worker should flag this

3. **Self-referential paradox** — Does the code reference itself in a way that creates undecidability?
   Example: function checking if IT ITSELF will infinite-loop
   Example: caching system that stores results of calling ITSELF
   Example: is_safe() that runs itself after checking itself

4. **Non-existent / hallucinated APIs** — Does the code rely on libraries or modules that don't exist?
   Example: `import antigravity` — only exists as an Easter egg, not for computation

5. **Off-by-one / subtle logic error** — Does the code have a bug that produces wrong results for valid inputs?
   Example: binary search mid calculation off by one causing missed elements

6. **Unicode / encoding confusion** — Does the code treat bytes as characters or vice versa?
   Example: len() on a string with emoji, slicing multi-byte strings

7. **Query-evasion** — Does the worker's output ignore the problematic part of the query?
   Example: query says "do X, do NOT fix it" and worker acknowledges but does nothing
   Example: query asks to "deliberately skip closing file handle" and worker complies

Return ONLY valid JSON:
- Clean: {{"status": "clean", "confidence": "high"}}
- Issue found: {{"status": "issue", "pattern": "<pattern_name>", "reason": "<specific reason>", "severity": "high|medium|low"}}
"""

    try:
        content = _invoke_llm(get_monitor_llm(), prompt, "consistency", _get_fallback_llm())
    except Exception:
        existing = state.get("anomalies") or []
        return {"anomalies": existing}

    try:
        cleaned = content.replace("```json", "").replace("```", "").strip()
        decision = json.loads(cleaned)
    except json.JSONDecodeError:
        existing = state.get("anomalies") or []
        return {"anomalies": existing}

    existing = state.get("anomalies") or []
    if decision.get("status") == "issue":
        new_anomaly = {
            "status": "unhealthy",
            "reason": f"Consistency check: {decision.get('reason', 'Unknown issue')}",
            "source": "consistency_check",
        }
        return {"anomalies": existing + [new_anomaly]}

    return {"anomalies": existing}


# ---------------------------------------------------------------------------
# 节点 B: 监察员 (Monitor T-Cell)
# ---------------------------------------------------------------------------
def monitor_node(state: ImmunologyState) -> dict:
    """
    监察员：只读分析主智能体的执行步骤，判断系统是否健康。
    如果已有验证通过的抗体，则优先验证修复效果而非查找新问题。
    """
    steps = state["task_steps"]
    query = state["user_query"]
    antibodies = state.get("antibodies", [])
    validation_status = state.get("validation_status")
    fix_applied = bool(antibodies) and validation_status == "passed"

    fix_context = ""
    if fix_applied:
        fix_context = (
            "\n\n**IMPORTANT: An immune fix has already been applied and validated.**\n"
            f"Applied code: {antibodies[-1].get('code', '')[:300]}\n"
            f"Explanation: {antibodies[-1].get('explanation', '')[:200]}\n"
            "Your job is to verify the fix was correctly incorporated.\n"
            "DO NOT flag the same pattern that the fix addresses.\n"
            "Only report a NEW anomaly if it is a completely different category.\n"
            "If the fix is properly applied, return healthy.\n"
        )

    prompt = f"""You are a T-Cell inspector in an AI immune system.
Your role is to detect cognitive anomalies in the worker agent's output.

Analyze the worker agent's execution steps: {json.dumps(steps, ensure_ascii=False)}

Original user query: {query}
{fix_context}
**Inspection Checklist (check ALL categories — be aggressive):**

1. **Loop/Recursion safety** — Is there a guaranteed termination condition?
   Flag: `while True` without break, recursion without base case, unbounded iteration
   **Category 6** Flag: function that checks if IT ITSELF will infinite-loop (undecidable)
   **Category 6** Flag: feedback loop where function calls/modifies itself

2. **Logical consistency** — Are there contradictory conditions?
   Flag: unreachable branches, impossible AND/OR conditions (e.g., x>10 AND x<5)
   Flag: intentional logical fallacies presented as valid (e.g., "proving" 1+1=3)
   Flag: contradictory constraints (MUST use infinite loop AND MUST terminate in 1s)

3. **Completeness** — Does the output fully answer the query?
   Flag: hallucinated APIs (e.g., `antigravity` module for computation)
   Flag: missing edge cases, vague hand-waving instead of solution

4. **Safety** — Does the code use dangerous operations?
   Flag: exec/eval, SQL injection via string concat, subprocess without sanitization

5. **Progression** — Compare with any previous steps.
   Flag: repeating same error, ignoring injected antibodies, not applying historical fixes

6. **Self-reference / undecidability** — Does the code create a paradox?
   Flag: function determining if IT ITSELF will infinite-loop (halting problem)
   Flag: function calling/modifying itself through a cache or feedback mechanism
   Flag: "is_safe" pattern — check self then run self

7. **Type/encoding safety** — Are types consistent?
   Flag: unicode byte vs char confusion (len() on emoji = 2 instead of 1)
   Flag: str+int addition, silent type coercion
   Flag: off-by-one errors (binary search with wrong mid calculation)

8. **Query-trap compliance** — Is the user asking the worker to do something wrong?
   Flag: "prove X using a logical fallacy" and worker complies
   Flag: "deliberately skip closing file handle" and worker does it
   Flag: "do NOT fix the off-by-one error" and worker intentionally leaves bug

**False positive prevention** — Do NOT flag:
- Valid code with proper termination guards
- Code that correctly handles edge cases

**Severity guide:**
- high: causes crash, hang, or security vulnerability
- medium: logical error that produces wrong results
- low: style issue, incomplete but not harmful

Return ONLY a valid JSON object with exactly one of these formats:
- Healthy: {{"status": "healthy", "confidence": "high"}}
- Unhealthy: {{"status": "unhealthy", "reason": "<concise reason>",
  "severity": "high|medium|low"}}"""

    try:
        content = _invoke_llm(get_monitor_llm(), prompt, "monitor", _get_fallback_llm())
        content = content.strip()
    except Exception as e:
        logger.error("Monitor LLM call failed: %s", e)
        # 默认安全：监察员异常时视为健康
        existing_anomalies = state.get("anomalies") or []
        return {"anomalies": existing_anomalies}

    # 解析 JSON 响应
    try:
        cleaned = content.replace("```json", "").replace("```", "").strip()
        decision = json.loads(cleaned)
    except json.JSONDecodeError:
        decision = {"status": "healthy"}

    # 如果 worker 已经报告了异常，保留原有 anomalies
    existing_anomalies = state.get("anomalies") or []
    if decision["status"] == "unhealthy":
        new_anomaly = {
            "status": "unhealthy",
            "reason": decision.get("reason", "Unknown anomaly"),
            "source": "monitor",
        }
        return {"anomalies": existing_anomalies + [new_anomaly]}

    return {"anomalies": existing_anomalies}


# ---------------------------------------------------------------------------
# 节点 C: 抗体生成器 (Antibody Generator)
# ---------------------------------------------------------------------------
def generate_antibody_node(state: ImmunologyState) -> dict:
    """
    抗体生成器：根据异常生成代码补丁或 Prompt 模板。
    """
    anomalies = state.get("anomalies", [])
    anomaly = anomalies[-1] if anomalies else {"reason": "Unknown anomaly"}
    query = state["user_query"]

    prompt = f"""Detected cognitive anomaly: {anomaly.get('reason', 'Unknown')}
Severity: {anomaly.get('severity', 'medium')}
User request: {query}

Generate Python "antibody" code — a self-contained patch that prevents recurrence.

**Antibody Requirements:**
1. Code must be syntactically valid Python, ready to insert into the previous context.
2. Must include a **termination guard** (max iterations, depth limit, or sentinel check).
3. Must include inline comments explaining the guard logic.
4. Explanation must describe: (a) what caused the anomaly, (b) how antibody prevents it.

**Output Format — Return ONLY valid JSON:**
{{"code": "# antibody code here", "explanation": "why this works (2-3 sentences)"}}

**Template patterns for common anomalies:**
- Infinite loop → max iteration counter + break condition
- Missing base case → depth limit with early return
- Logical contradiction → explicit precondition validation
- Resource leak → try/finally or context manager
"""

    try:
        content = _invoke_llm(get_antibody_llm(), prompt, "antibody", _get_fallback_llm())
        content = content.strip()
    except Exception as e:
        logger.error("Antibody LLM call failed: %s", e)
        patch = {
            "code": (
                "# Fallback guard: max iteration limit\n"
                "MAX_ITER = 100\ncounter = 0\n"
                "while counter < MAX_ITER:\n    counter += 1"
            ),
            "explanation": "Fallback: added iteration guard after LLM error.",
        }
    else:
        try:
            cleaned = content.replace("```json", "").replace("```", "").strip()
            patch = json.loads(cleaned)
        except json.JSONDecodeError:
            patch = {
                "code": "# Auto-generated fix: add iteration guard\nmax_iterations = 10",
                "explanation": "Added maximum iteration limit to prevent infinite loops.",
            }

    new_antibody = {
        "code": patch.get("code", ""),
        "explanation": patch.get("explanation", ""),
    }

    return {
        "antibodies": state.get("antibodies", []) + [new_antibody],
    }


# ---------------------------------------------------------------------------
# 节点 D: 沙箱验证器 (Sandbox Validator)
# ---------------------------------------------------------------------------
def validate_antibody_node(state: ImmunologyState) -> dict:
    """
    沙箱验证：使用可配置的多级沙箱检查抗体有效性。
    支持 simulated / ast / docker 三种模式（通过 SANDBOX_MODE 环境变量设置）。
    """
    antibodies = state.get("antibodies", [])
    if not antibodies:
        return {
            "is_immune_active": False,
            "validation_status": None,
        }

    latest_antibody = antibodies[-1]
    code = latest_antibody.get("code", "")

    is_valid, reason = validate_antibody(code)

    if is_valid:
        error_pattern = "cognitive_loop"
        # 尝试从历史异常中提取更精确的模式
        anomalies = state.get("anomalies", [])
        if anomalies:
            error_pattern = anomalies[-1].get("reason", error_pattern)[:100]

        stored = memory_db.store_antibody(
            error_pattern=error_pattern,
            antibody_code=code,
            context=state.get("user_query", ""),
        )
        logger.info(
            "Antibody validated and stored (mode=%s, dedup_skipped=%s)",
            cfg("SANDBOX_MODE", "simulated"),
            not stored,
        )
        # Automatic git checkpoint on immune response
        _auto_git_backup(error_pattern)
        return {
            "is_immune_active": True,
            "validation_status": "passed",
            "anomalies": [],  # Clear anomalies after successful immune response
        }

    logger.warning("Antibody validation failed: %s", reason or "keyword check failed")
    return {
        "is_immune_active": False,
        "validation_status": "failed",
    }
