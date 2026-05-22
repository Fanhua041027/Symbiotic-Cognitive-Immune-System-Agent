"""LangGraph 工作流定义，将各节点连接为闭环免疫系统。"""

from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

from core.logger import setup_logger
from core.state import ImmunologyState
from core.escalation import escalation
from core.config import get as cfg
from core.nodes import (
    main_worker_node,
    monitor_node,
    generate_antibody_node,
    validate_antibody_node,
)

# 路由类型
Route = Literal["continue", "immune_response", "end"]


logger = setup_logger("workflow")


# ---------------------------------------------------------------------------
# 执行轨迹装饰器 - 自动为每个节点注入 trace 信息
# ---------------------------------------------------------------------------
_TRACE_LABELS = {
    "worker": "Worker",
    "monitor": "Monitor T-Cell",
    "generate_antibody": "Antibody Generator",
    "validate_antibody": "Sandbox Validator",
}


def _with_trace(node_name: str, func):
    """Wrap a node function to inject workflow trace."""
    label = _TRACE_LABELS.get(node_name, node_name)

    def wrapped(state: ImmunologyState) -> dict:
        trace = list(state.get("workflow_trace") or [])
        trace.append(f"enter:{node_name}")
        logger.debug("Trace: entering %s (iter=%d)", label, state.get("iteration_count", 0))

        result = func(state)

        # Preserve trace in return dict without mutating input state
        if isinstance(result, dict) and "workflow_trace" not in result:
            result["workflow_trace"] = trace
        return result

    return wrapped


# ---------------------------------------------------------------------------
# Finalize 节点 — 在流程结束前处理升级/成功记录
# ---------------------------------------------------------------------------
def finalize_node(state: ImmunologyState) -> dict:
    """
    最终节点：处理升级报告和成功记录，不参与路由决策。

    在流程结束时运行，负责记录免疫响应的结果。
    """
    result: dict = {}
    max_iterations = cfg("MAX_ITERATIONS", 5)
    anomalies = state.get("anomalies", [])
    has_anomalies = bool(anomalies)
    has_output = state.get("final_output") is not None
    iteration = state.get("iteration_count") or 0
    has_antibodies = bool(state.get("antibodies"))

    if iteration >= max_iterations:
        logger.warning("Max iterations (%d) reached.", max_iterations)
        if has_anomalies:
            report_path = escalation.record_failure(
                query=state.get("user_query", ""),
                anomaly_reason=(
                    anomalies[-1].get("reason", "Unknown") if anomalies else "Unknown"
                ),
                antibodies_generated=len(state.get("antibodies", [])),
            )
            if report_path:
                result["escalation_report"] = report_path
        elif has_output and has_antibodies:
            escalation.record_success()
    elif has_output and has_antibodies:
        escalation.record_success()

    return result


# ---------------------------------------------------------------------------
# 条件路由 (pure routing, no side effects)
# ---------------------------------------------------------------------------
def should_continue(state: ImmunologyState) -> Route:
    """
    路由决策：根据当前状态决定下一步走向。

    决策逻辑:
      - 超过最大迭代次数 → 结束（防止无限循环）
      - 有未处理的异常 → 免疫响应 (生成抗体)
      - 已有最终输出且健康 → 结束
      - 否则 → 继续执行

    Note: 本函数仅做路由判断，不修改 state。
          升级/成功记录由 finalize_node 处理。
    """
    max_iterations = cfg("MAX_ITERATIONS", 5)
    anomalies = state.get("anomalies", [])
    has_anomalies = bool(anomalies)
    has_output = state.get("final_output") is not None
    iteration = state.get("iteration_count") or 0

    decision: Route
    if iteration >= max_iterations:
        decision = "end"
    elif has_anomalies:
        decision = "immune_response"
    elif has_output:
        decision = "end"
    else:
        decision = "continue"

    logger.debug("Route: %s (iter=%d, anomalies=%d, output=%s)",
                  decision, iteration, len(anomalies), has_output)
    return decision


def build_workflow() -> StateGraph:
    """构建并编译完整的免疫智能体工作流图。"""

    workflow = StateGraph(ImmunologyState)

    # 注册节点 (用 trace wrapper 包装)
    workflow.add_node("worker", _with_trace("worker", main_worker_node))
    workflow.add_node("monitor", _with_trace("monitor", monitor_node))
    workflow.add_node("generate_antibody", _with_trace("generate_antibody", generate_antibody_node))
    workflow.add_node("validate_antibody", _with_trace("validate_antibody", validate_antibody_node))
    workflow.add_node("finalize", _with_trace("finalize", finalize_node))

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
            "end": "finalize",
        },
    )

    # 免疫回路: 生成抗体 → 验证 → 回到主智能体重试
    workflow.add_edge("generate_antibody", "validate_antibody")
    workflow.add_edge("validate_antibody", "worker")

    # 终结点：finalize → END
    workflow.add_edge("finalize", END)

    return workflow.compile()  # type: ignore[return-value]


# 编译好的应用实例
app = build_workflow()
