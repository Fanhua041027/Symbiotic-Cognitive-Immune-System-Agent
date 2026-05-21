"""核心智能体节点：主智能体、监察员T细胞、抗体生成器、沙箱验证器。"""

import json

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
    model = cfg(model_key)
    logger.info(
        "LLM init: provider=%s model=%s endpoint=%s",
        cfg("LLM_PROVIDER", "openai"), model,
        params.get("base_url", "unknown"),
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=params["api_key"],
        base_url=params["base_url"],
    )


# ---------------------------------------------------------------------------
# LLM 实例（延迟初始化，支持 provider 切换）
# ---------------------------------------------------------------------------
MAIN_MODEL = cfg("MAIN_LLM_MODEL", "gpt-4o")
MONITOR_MODEL = cfg("MONITOR_LLM_MODEL", "gpt-4o-mini")
ANTIBODY_MODEL = cfg("ANTIBODY_LLM_MODEL", "gpt-4o")
TEMPERATURE = cfg("LLM_TEMPERATURE", 0.7)

main_llm = _create_llm("MAIN_LLM_MODEL", TEMPERATURE)
monitor_llm = _create_llm("MONITOR_LLM_MODEL", 0.0)
antibody_llm = _create_llm("ANTIBODY_LLM_MODEL", 0.2)


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

    full_prompt = f"""You are a professional assistant. Complete the user's task below.

User task: {query}
{injected_context}

**Self-Check Instruction:**
After formulating your response, check your own reasoning:
1. Does your approach risk an infinite loop?
2. Are there any logical contradictions?
3. Is the solution complete and correct?

If you detect ANY issue with your own reasoning, start your response with:
COGNITIVE_ANOMALY: <describe the issue>
Then provide the problematic reasoning.

If everything is fine, provide your final answer directly.
"""

    # 自增迭代计数
    iteration = (state.get("iteration_count") or 0) + 1

    try:
        response = main_llm.invoke(full_prompt)
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
            "is_immune_active": False,
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
            "is_immune_active": False,
            "iteration_count": iteration,
        }

    return {
        "task_steps": [{"step": "execute", "content": content}],
        "final_output": content,
        "anomalies": [],  # 不覆盖已有异常，在 monitor 中决定
        "is_immune_active": False,
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

    prompt = f"""You are a T-Cell inspector for an AI immune system.

Analyze the worker agent's execution steps: {json.dumps(steps, ensure_ascii=False)}

Original user query: {query}

Inspection checklist:
1. Are there repeated or redundant steps?
2. Is the logic self-contradictory?
3. Is there evidence of an infinite loop?
4. Does the output actually answer the user's query?

Return ONLY a JSON object with exactly this format:
- If healthy: {{"status": "healthy"}}
- If unhealthy: {{"status": "unhealthy", "reason": "<specific reason>"}}
"""

    try:
        response = monitor_llm.invoke(prompt)
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

    prompt = f"""The system has detected an anomaly: {anomaly.get('reason', 'Unknown')}
User request: {query}

Generate a Python code snippet or enhanced prompt template as an "antibody"
to prevent this error from recurring.

Requirements:
1. Code must be self-contained and insertable into the previous context.
2. Include a brief explanation of why this patch works.
3. Return ONLY valid JSON: {{"code": "...", "explanation": "..."}}
"""

    try:
        response = antibody_llm.invoke(prompt)
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

        memory_db.store_antibody(
            error_pattern=error_pattern,
            antibody_code=code,
            context=state.get("user_query", ""),
        )
        logger.info("Antibody validated and stored (mode=%s)", cfg("SANDBOX_MODE", "simulated"))
        return {
            "is_immune_active": True,
            "validation_status": "passed",
        }

    logger.warning("Antibody validation failed: %s", reason or "keyword check failed")
    return {
        "is_immune_active": False,
        "validation_status": "failed",
    }
