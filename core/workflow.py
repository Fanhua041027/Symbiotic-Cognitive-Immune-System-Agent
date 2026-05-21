"""LangGraph 工作流定义，将各节点连接为闭环免疫系统。"""

import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

from core.logger import setup_logger
from core.state import ImmunologyState
from core.escalation import escalation
from core.nodes import (
    main_worker_node,
    monitor_node,
    generate_antibody_node,
    validate_antibody_node,
)

# 路由类型
Route = Literal["continue", "immune_response", "end"]


logger = setup_logger("workflow")

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))


def should_continue(state: ImmunologyState) -> Route:
    """
    路由决策：根据当前状态决定下一步走向。

    决策逻辑:
      - 超过最大迭代次数 → 结束（防止无限循环）
      - 有未处理的异常 → 免疫响应 (生成抗体)
      - 已有最终输出且健康 → 结束
      - 否则 → 继续执行
    """
    anomalies = state.get("anomalies", [])
    has_anomalies = bool(anomalies)
    has_output = state.get("final_output") is not None
    iteration = state.get("iteration_count") or 0

    if iteration >= MAX_ITERATIONS:
        logger.warning("Max iterations (%d) reached, forcing end.", MAX_ITERATIONS)
        if has_anomalies:
            report_path = escalation.record_failure(
                query=state.get("user_query", ""),
                anomaly_reason=(
                    state["anomalies"][-1].get("reason", "Unknown")
                    if state.get("anomalies") else "Unknown"
                ),
                antibodies_generated=len(state.get("antibodies", [])),
            )
            if report_path:
                state["escalation_report"] = report_path
        return "end"
    if has_anomalies:
        return "immune_response"
    elif has_output:
        # 如果之前产生过抗体（免疫系统曾介入），记录恢复成功
        if state.get("antibodies"):
            escalation.record_success()
        return "end"
    else:
        return "continue"


def build_workflow() -> StateGraph:
    """构建并编译完整的免疫智能体工作流图。"""

    workflow = StateGraph(ImmunologyState)

    # 注册节点
    workflow.add_node("worker", main_worker_node)
    workflow.add_node("monitor", monitor_node)
    workflow.add_node("generate_antibody", generate_antibody_node)
    workflow.add_node("validate_antibody", validate_antibody_node)

    # 入口 → 主智能体
    workflow.add_edge(START, "worker")

    # 主智能体 → 监察员
    workflow.add_edge("worker", "monitor")

    # 监察员 → 条件路由
    workflow.add_conditional_edges(
        "monitor",
        should_continue,
        {
            "continue": "worker",
            "immune_response": "generate_antibody",
            "end": END,
        },
    )

    # 免疫回路: 生成抗体 → 验证 → 回到主智能体重试
    workflow.add_edge("generate_antibody", "validate_antibody")
    workflow.add_edge("validate_antibody", "worker")

    return workflow.compile()


# 编译好的应用实例
app = build_workflow()
