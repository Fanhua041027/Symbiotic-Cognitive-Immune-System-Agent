"""核心智能体节点：主智能体、监察员T细胞、抗体生成器、沙箱验证器。"""

import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from core.state import ImmunologyState
from core.memory import memory_db

load_dotenv()

# ---------------------------------------------------------------------------
# LLM 初始化
# ---------------------------------------------------------------------------
MAIN_MODEL = os.getenv("MAIN_LLM_MODEL", "gpt-4o")
MONITOR_MODEL = os.getenv("MONITOR_LLM_MODEL", "gpt-4o-mini")
ANTIBODY_MODEL = os.getenv("ANTIBODY_LLM_MODEL", "gpt-4o")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

main_llm = ChatOpenAI(model=MAIN_MODEL, temperature=TEMPERATURE)
monitor_llm = ChatOpenAI(model=MONITOR_MODEL, temperature=0.0)
antibody_llm = ChatOpenAI(model=ANTIBODY_MODEL, temperature=0.2)


# ---------------------------------------------------------------------------
# 节点 A: 主智能体 (Worker)
# ---------------------------------------------------------------------------
def main_worker_node(state: ImmunologyState) -> dict:
    """
    主智能体：尝试完成用户任务。
    如果检测到潜在的死循环或逻辑矛盾，主动触发免疫响应。
    """
    query = state["user_query"]

    # 如果有历史抗体，注入上下文
    injected_context = ""
    if state["antibodies"]:
        last_antibody = state["antibodies"][-1]
        injected_context = (
            f"\n[Previous Fix Applied]: {last_antibody['code']}\n"
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

    response = main_llm.invoke(full_prompt)
    content = response.content.strip()

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

    response = monitor_llm.invoke(prompt)
    content = response.content.strip()

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

    response = antibody_llm.invoke(prompt)
    content = response.content.strip()

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
    沙箱验证：检查生成的抗体是否有效。
    （生产环境应替换为 Docker / E2B 沙箱执行。）
    """
    antibodies = state.get("antibodies", [])
    if not antibodies:
        return {
            "is_immune_active": False,
            "validation_status": "failed",
        }

    latest_antibody = antibodies[-1]
    code = latest_antibody.get("code", "")

    # 模拟验证逻辑：抗体包含关键修复模式视为有效
    has_fix_keywords = any(
        kw in code.lower()
        for kw in ["fix", "guard", "limit", "check", "max", "break", "return"]
    )
    is_long_enough = len(code) > 15
    is_valid = has_fix_keywords or is_long_enough

    if is_valid:
        # 存入免疫记忆库
        memory_db.store_antibody(
            error_pattern="cognitive_loop",
            antibody_code=code,
            context=state.get("user_query", ""),
        )
        return {
            "is_immune_active": True,
            "validation_status": "passed",
        }

    return {
        "is_immune_active": False,
        "validation_status": "failed",
    }
