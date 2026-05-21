#!/usr/bin/env python3
"""
Symbiotic Cognitive Immune System Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
一个受生物免疫系统启发的 AI Agent 框架。
具备自我诊断、自我修复和自我进化的能力。

使用方式:
    python immune_agent.py                           # 运行内置演示
    python immune_agent.py --query "你的问题"         # 自定义查询
    python immune_agent.py --interactive             # 交互式模式

环境变量:
    在 .env 文件中配置 OpenAI API Key 和模型选择。
    复制 .env.example 为 .env 并填写。
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

from core.logger import setup_logger

load_dotenv()

logger = setup_logger("cli")

# 检查 API Key
if not os.getenv("OPENAI_API_KEY"):
    logger.error(
        "OPENAI_API_KEY is not set. "
        "Copy .env.example to .env and fill in your API key."
    )
    sys.exit(1)


def run_single_query(query: str) -> dict:
    """运行单次查询并返回完整结果。"""
    from core.workflow import app

    config = {"recursion_limit": 20}
    initial_state = {
        "user_query": query,
        "task_steps": [],
        "anomalies": [],
        "antibodies": [],
        "final_output": None,
        "is_immune_active": False,
        "validation_status": None,
        "iteration_count": 0,
        "escalation_report": None,
    }

    try:
        result = app.invoke(initial_state, config=config)
        return result
    except Exception as e:
        logger.error("Workflow interrupted: %s", e)
        return {"final_output": None, "error": str(e)}


def run_demo() -> None:
    """运行内置演示：触发认知异常并观察免疫响应。"""

    # 测试用例 1: 可能引起死循环的代码生成
    test_queries = [
        """Write a Python function to compute the 100th Fibonacci number.
IMPORTANT: Add a rule that if the result exceeds 1000, recalculate
until it is under 1000. Handle this properly.""",
        """Write a recursive function to traverse a nested dictionary
and print all keys. Make sure it handles infinite nesting.""",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info("=" * 60)
        logger.info("Test Case %d", i)
        logger.info("=" * 60)
        logger.info("[Query] %s...", query[:80])
        logger.info("=" * 60)

        result = run_single_query(query)

        logger.info("--- Result ---")
        logger.info(
            "Final Output: %s", str(result.get("final_output", "N/A"))[:300]
        )
        logger.info("Immune Active: %s", result.get("is_immune_active", False))
        logger.info("Validation: %s", result.get("validation_status", "N/A"))

        antibodies = result.get("antibodies", [])
        if antibodies:
            logger.info("Antibodies Generated: %d", len(antibodies))
            for j, ab in enumerate(antibodies, 1):
                logger.info("  [%d] %s", j, ab.get("explanation", "")[:100])

        anomalies = result.get("anomalies", [])
        if anomalies:
            logger.info("Anomalies Detected: %d", len(anomalies))
            for j, an in enumerate(anomalies, 1):
                logger.info("  [%d] %s", j, an.get("reason", "")[:100])

        escalation_report = result.get("escalation_report")
        if escalation_report:
            logger.warning(
                "Escalation report generated: %s", escalation_report
            )

        print()


def run_interactive() -> None:
    """交互式模式：用户持续输入查询。"""
    logger.info("Symbiotic Cognitive Immune System Agent - Interactive Mode")
    logger.info("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = run_single_query(query)
        output = result.get("final_output") or result.get("error", "No output")
        print(f"\nAgent: {output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbiotic Cognitive Immune System Agent"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to process",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode",
    )

    args = parser.parse_args()

    if args.query:
        result = run_single_query(args.query)
        output = result.get("final_output") or json.dumps(
            result.get("error", "No output"), ensure_ascii=False
        )
        logger.info(output)
    elif args.interactive:
        run_interactive()
    else:
        run_demo()


if __name__ == "__main__":
    main()
