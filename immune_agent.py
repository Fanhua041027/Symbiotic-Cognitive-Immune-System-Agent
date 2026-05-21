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
from core.config import validate_all, show_summary

load_dotenv()

logger = setup_logger("cli")

# 启动时校验配置
config_warnings = validate_all()
show_summary()

if config_warnings:
    has_critical = any(w.startswith("MISSING: OPENAI_API_KEY") for w in config_warnings)
    if has_critical:
        # Only bail for commands that need the API key
        if len(sys.argv) > 1 and sys.argv[1] in ("-s", "--stats", "-g", "--graph"):
            pass  # stats/graph don't need API key
        else:
            logger.error(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your API key."
            )
            sys.exit(1)


def run_single_query(query: str, timeout: float = 60.0) -> dict:
    """运行单次查询并返回完整结果。

    Args:
        query: 用户输入的问题
        timeout: 超时秒数（默认 60s，超过返回 timeout 错误）
    """
    import signal

    from core.workflow import app

    config = {"recursion_limit": max(20, cfg("MAX_ITERATIONS", 5) * 4)}
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

    class TimeoutError_(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError_(f"Query timed out after {timeout}s")

    original_handler = None
    if hasattr(signal, "SIGALRM"):
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout))

    try:
        result = app.invoke(initial_state, config=config)
        return result
    except TimeoutError_ as e:
        logger.error("Workflow timed out: %s", e)
        return {"final_output": None, "error": str(e)}
    except Exception as e:
        logger.error("Workflow interrupted: %s", e)
        return {"final_output": None, "error": str(e)}
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if original_handler:
                signal.signal(signal.SIGALRM, original_handler)


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


def run_benchmark() -> None:
    """Run the adversarial benchmark suite."""
    from tests.adversarial import run_benchmark as _bench

    logger.info("Starting adversarial benchmark...")
    _bench()


def show_stats() -> None:
    """Display immune memory and system statistics."""
    from core.memory import memory_db
    from core.config import get as cfg_get

    print("\n" + "=" * 50)
    print("  Immune System Statistics")
    print("=" * 50)

    # Immune memory stats
    try:
        count = memory_db.count()
        backend = getattr(memory_db, '_backend', 'unknown')
        print(f"\n  Immune Memory ({backend}):")
        print(f"    Stored Antibodies : {count}")
    except Exception as e:
        print(f"\n  Immune Memory: error reading ({e})")

    # Config summary
    print(f"\n  Configuration:")
    print(f"    Provider       : {cfg_get('LLM_PROVIDER', 'openai')}")
    print(f"    Worker Model   : {cfg_get('MAIN_LLM_MODEL', 'gpt-4o')}")
    print(f"    Monitor Model  : {cfg_get('MONITOR_LLM_MODEL', 'gpt-4o-mini')}")
    print(f"    Sandbox Mode   : {cfg_get('SANDBOX_MODE', 'simulated')}")
    print(f"    Max Iterations : {cfg_get('MAX_ITERATIONS', 5)}")
    print(f"    Escalation Thr : {cfg_get('ESCALATION_THRESHOLD', 3)}")

    print(f"\n  Project Info:")
    print(f"    Path   : {os.path.dirname(os.path.abspath(__file__))}")
    print(f"    Logs   : {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')}")
    print(f"    Memory : {os.path.join(os.path.dirname(os.path.abspath(__file__)), '.immune_db')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbiotic Cognitive Immune System Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                        Run demo test cases\n"
            "  %(prog)s -q \"write a function\"  Single query\n"
            "  %(prog)s -i                      Interactive mode\n"
            "  %(prog)s -b                      Run adversarial benchmark\n"
            "  %(prog)s -g                      Show workflow graph\n"
            "  %(prog)s -s                      Show immune memory stats\n"
            "  %(prog)s -q \"hello\" -j          Output as JSON\n"
            "  %(prog)s -q \"hello\" -t 30      Query with 30s timeout\n"
        ),
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
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run adversarial benchmark suite",
    )
    parser.add_argument(
        "--graph", "-g",
        action="store_true",
        help="Print workflow graph visualization",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON (use with --query)",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=60.0,
        help="Query timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show immune memory statistics",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.graph:
        from core.viz import print_graph, print_graph_ascii
        print_graph()
        print_graph_ascii()
        return

    if args.benchmark:
        run_benchmark()
        return

    if args.query:
        result = run_single_query(args.query, timeout=args.timeout)
        if args.json:
            print(json.dumps(
                {k: v for k, v in result.items()
                 if k in ("final_output", "anomalies", "antibodies",
                          "is_immune_active", "validation_status",
                          "escalation_report")},
                ensure_ascii=False, indent=2, default=str,
            ))
        else:
            output = result.get("final_output") or result.get("error", "No output")
            logger.info(output)
        return

    if args.interactive:
        run_interactive()
        return

    run_demo()


if __name__ == "__main__":
    main()
