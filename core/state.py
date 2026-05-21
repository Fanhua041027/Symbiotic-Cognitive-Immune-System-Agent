"""State schema definitions for the immune agent workflow."""

from typing import TypedDict, List, Dict, Any, Optional


class ImmunologyState(TypedDict):
    """核心状态结构，贯穿整个免疫智能体工作流。"""

    # 用户原始指令
    user_query: str

    # 当前任务执行步骤列表
    task_steps: List[Dict[str, Any]]

    # 异常检测记录
    anomalies: List[Dict[str, Any]]

    # 生成的抗体列表 (补丁)
    antibodies: List[Dict[str, Any]]

    # 最终输出结果
    final_output: Optional[str]

    # 系统标记：是否进入免疫模式
    is_immune_active: bool

    # 抗体验证状态
    validation_status: Optional[str]

    # 迭代计数 (防止死循环)
    iteration_count: int

    # 告警升级报告路径 (当免疫系统无法自主修复时)
    escalation_report: Optional[str]

    # 工作流执行轨迹 (用于可视化)
    workflow_trace: List[str]
