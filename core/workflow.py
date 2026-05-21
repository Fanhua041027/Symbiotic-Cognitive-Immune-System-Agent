"""LangGraph 工作流定义，将各节点连接为闭环免疫系统。"""

import os
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

from core.state import ImmunologyState
from core.nodes import (
    main_worker_node,
    monitor_node,
    generate_antibody_node,
    validate_antibody_node,
)

# 路由类型
Route = Literal["continue", "immune_response", "end"]


MAX_ITERATIONS = 5  # 最大免疫迭代次数，防止无限修复循环


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
        print(f"[Workflow] Max iterations ({MAX_ITERATIONS}) reached, forcing end.")
        return "end"
    if has_anomalies:
        return "immune_response"
    elif has_output:
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
