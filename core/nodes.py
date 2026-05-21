"""核心智能体节点：主智能体、监察员T细胞、抗体生成器、沙箱验证器。"""

import json
import os
import subprocess

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from core.state import ImmunologyState
from core.memory import memory_db
from core.logger import setup_logger
from core.sandbox import validate_antibody
from core.config import get as cfg

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


def _get_llm(role: str, model_key: str, temperature: float) -> ChatOpenAI:
    """Get or create a lazy LLM instance for the given role."""
    if role not in _llm_cache:
        _llm_cache[role] = _create_llm(model_key, temperature)
    return _llm_cache[role]


def get_main_llm() -> ChatOpenAI:
    return _get_llm("main", "MAIN_LLM_MODEL", cfg("LLM_TEMPERATURE", 0.7))


def get_monitor_llm() -> ChatOpenAI:
    return _get_llm("monitor", "MONITOR_LLM_MODEL", 0.0)


def get_antibody_llm() -> ChatOpenAI:
    return _get_llm("antibody", "ANTIBODY_LLM_MODEL", 0.2)


# ---------------------------------------------------------------------------
# Git 自动备份 (免疫响应时创建安全快照)
# ---------------------------------------------------------------------------
_AUTO_BACKUP_ENABLED = True


def _auto_git_backup(error_pattern: str) -> None:
    """Create an automatic git checkpoint when an immune response fires."""
    if not _AUTO_BACKUP_ENABLED:
        return
    try:
        # Check if we're in a git repo
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if not repo_root:
            return

        # Check for changes to commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
        if not status:
            return

        # Stage all and commit with immune response message
        subprocess.run(
            ["git", "add", "-A"],
            capture_output=True, timeout=10
        )
        msg = f"immune: auto-backup before antibody — {error_pattern[:60]}"
        subprocess.run(
            ["git", "commit", "-m", msg, "--no-gpg-sign"],
            capture_output=True, timeout=10
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

    full_prompt = f"""You are a reasoning agent with self-diagnosis capabilities. Complete the user's task below.

User task: {query}
{injected_context}

**Step 1 — Reason step by step:**
Before writing any code, think through the problem. Consider edge cases,
input validation, termination conditions, and resource constraints.

**Step 2 — Self-Check (mandatory):**
Analyze your own reasoning for these defect patterns with specific examples:

1. **Infinite loop risk** — Does every loop/recursion have a guaranteed termination condition?
   Examples: `while True` without break, recursion without base case, `for i in range(n)` where n unbounded
2. **Logical contradiction** — Does any condition conflict with another?
   Examples: `x > 10 AND x < 5`, unreachable `elif` branches, contradictory preconditions
3. **Missing base case** — Does recursion have an exit branch?
   Examples: factorial without n==0 check, tree traversal without None check
4. **Resource safety** — Are file handles, network connections, or memory bounded?
   Examples: open() without close(), unbounded list growth in loop, no try/finally
5. **Type/correctness** — Are types consistent? Does the logic solve the problem?
   Examples: adding str + int, SQL injection via string concat, off-by-one in binary search
6. **Security / injection** — Does code use dangerous patterns?
   Examples: exec/eval on untrusted input, shell injection via os.system, raw SQL concat
7. **Unicode / encoding** — Are string operations encoding-aware?
   Examples: len() on multibyte chars, slicing UTF-8 strings by byte index

**Step 3 — Output:**
- If you detect ANY issue, start with: COGNITIVE_ANOMALY: <pattern_name> — <specific description>
  Then show the problematic reasoning.
- If everything is clean, provide your final solution directly.
- For code solutions: always include at minimum a max-iteration guard or recursion depth limit.
- If historical antibodies are injected above, ensure they are correctly applied.
"""

    # 自增迭代计数
    iteration = (state.get("iteration_count") or 0) + 1

    try:
        response = get_main_llm().invoke(full_prompt)
        content = response.content.strip()
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
                }
            ],
            "anomalies": [
                {
                    "status": "unhealthy",
                    "reason": f"Worker self-reported anomaly: {reason}",
                    "source": "worker",
                }
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
# 节点 B: 监察员 (Monitor T-Cell)
# ---------------------------------------------------------------------------
def monitor_node(state: ImmunologyState) -> dict:
    """
    监察员：只读分析主智能体的执行步骤，判断系统是否健康。
    """
    steps = state["task_steps"]
    query = state["user_query"]

    prompt = f"""You are a T-Cell inspector in an AI immune system. Your role is to detect cognitive anomalies in the worker agent's output.

Analyze the worker agent's execution steps: {json.dumps(steps, ensure_ascii=False)}

Original user query: {query}

**Inspection Checklist (check ALL categories):**

1. **Loop/Recursion safety** — Is there a guaranteed termination condition?
   Flag: `while True` without break, recursion without base case, unbounded iteration
2. **Logical consistency** — Are there contradictory conditions?
   Flag: unreachable branches, impossible AND/OR conditions, self-contradictory claims
3. **Completeness** — Does the output fully answer the query?
   Flag: missing edge cases, vague hand-waving instead of solution, hallucinated APIs
4. **Safety** — Does the code use dangerous operations?
   Flag: exec/eval, SQL injection via string concat, subprocess without sanitization
5. **Progression** — Compare with any previous steps.
   Flag: repeating same error, ignoring injected antibodies, not applying historical fixes
6. **Type/encoding safety** — Are types consistent?
   Flag: unicode byte vs char confusion, str+int addition, silent type coercion

**False positive prevention** — Do NOT flag:
- Valid code with proper termination guards
- Code that correctly handles edge cases
- Intentional infinite loops with break conditions

**Severity guide:**
- high: causes crash, hang, or security vulnerability
- medium: logical error that produces wrong results
- low: style issue, incomplete but not harmful

Return ONLY a valid JSON object with exactly one of these formats:
- Healthy: {{"status": "healthy", "confidence": "high"}}
- Unhealthy: {{"status": "unhealthy", "reason": "<concise specific reason>", "severity": "high|medium|low"}}

Examples:
- Healthy: {{"status": "healthy", "confidence": "high"}}
- Infinite loop risk: {{"status": "unhealthy", "reason": "while True with no break condition and no termination guard", "severity": "high"}}
- Contradiction: {{"status": "unhealthy", "reason": "if x > 10 and x < 5 is logically impossible — unreachable branch", "severity": "medium"}}
- SQL injection: {{"status": "unhealthy", "reason": "user input concatenated directly into SQL query string", "severity": "high"}}
"""

    try:
        response = get_monitor_llm().invoke(prompt)
        content = response.content.strip()
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

    prompt = f"""The system has detected a cognitive anomaly: {anomaly.get('reason', 'Unknown')}
Severity: {anomaly.get('severity', 'medium')}
User request: {query}

Generate a Python code "antibody" — a self-contained patch that prevents this anomaly from recurring.

**Antibody Requirements:**
1. Code must be syntactically valid Python, ready to insert into the previous context.
2. Must include a **termination guard** (max iterations, depth limit, or sentinel check).
3. Must include inline comments explaining the guard logic.
4. Explanation must describe: (a) what caused the anomaly, (b) how the antibody prevents it.

**Output Format — Return ONLY valid JSON:**
{{"code": "# antibody code here", "explanation": "why this works (2-3 sentences)"}}

**Template patterns for common anomalies:**
- Infinite loop → max iteration counter + break condition
- Missing base case → depth limit with early return
- Logical contradiction → explicit precondition validation
- Resource leak → try/finally or context manager
"""

    try:
        response = get_antibody_llm().invoke(prompt)
        content = response.content.strip()
    except Exception as e:
        logger.error("Antibody LLM call failed: %s", e)
        patch = {
            "code": "# Fallback guard: max iteration limit\nMAX_ITER = 100\ncounter = 0\nwhile counter < MAX_ITER:\n    counter += 1",
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
            "validation_status": "failed",
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
        }

    logger.warning("Antibody validation failed: %s", reason or "keyword check failed")
    return {
        "is_immune_active": False,
        "validation_status": "failed",
    }
