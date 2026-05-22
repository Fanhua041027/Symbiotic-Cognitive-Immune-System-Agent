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

from core.config import get as cfg
from core.config import show_summary, validate_all
from core.logger import setup_logger

load_dotenv()

logger = setup_logger("cli")

# 启动时校验配置（仅在直接执行时退出，import 时只警告）
config_warnings = validate_all()
show_summary()

if config_warnings:
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "deepseek":
        has_critical = any(w.startswith("MISSING: DEEPSEEK_API_KEY") for w in config_warnings)
    elif provider == "custom":
        has_critical = any(w.startswith("MISSING: CUSTOM_API_KEY") for w in config_warnings)
    else:
        has_critical = any(w.startswith("MISSING: OPENAI_API_KEY") for w in config_warnings)
    if has_critical and __name__ == "__main__":
        # Only bail for commands that need the API key
        if len(sys.argv) > 1 and sys.argv[1] in ("-s", "--stats", "-g", "--graph"):
            pass  # stats/graph don't need API key
        else:
            logger.error(
                f"{'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'CUSTOM_API_KEY' if provider == 'custom' else 'OPENAI_API_KEY'} is not set. "
                "Copy .env.example to .env and fill in your API key."
            )
            sys.exit(1)


def run_single_query(
    query: str,
    timeout: float = 60.0,
    record_session: bool = True,
) -> dict:
    """运行单次查询并返回完整结果。

    Args:
        query: 用户输入的问题
        timeout: 超时秒数
        record_session: 是否记录到会话管理
    """
    import concurrent.futures
    import signal
    import time as _time

    from core.agent_session import get_session
    from core.escalation import escalation
    from core.metrics import metrics
    from core.state import ImmunologyState
    from core.workflow import app

    # Reset per-query state to prevent cross-query bleeding
    escalation.reset()

    config = {"recursion_limit": max(20, cfg("MAX_ITERATIONS", 5) * 4)}
    initial_state: ImmunologyState = {
        "user_query": query,
        "task_steps": [],
        "anomalies": [],
        "antibodies": [],
        "final_output": None,
        "is_immune_active": False,
        "validation_status": None,
        "iteration_count": 0,
        "escalation_report": None,
        "workflow_trace": [],
    }

    class TimeoutError_(Exception):
        pass

    start_time = _time.time()

    # Use SIGALRM on Unix, threading fallback on Windows
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        def _timeout_handler(signum, frame):
            raise TimeoutError_(f"Query timed out after {timeout}s")
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler)  # type: ignore[attr-defined]
        signal.alarm(int(timeout))  # type: ignore[attr-defined]

    try:
        if use_alarm:
            result = app.invoke(initial_state, config=config)  # type: ignore[attr-defined]
        else:
            # Windows: use ThreadPoolExecutor for timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(app.invoke, initial_state, config)  # type: ignore[attr-defined]
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError_(f"Query timed out after {timeout}s")

        duration = _time.time() - start_time
        result["user_query"] = query
        result["duration"] = duration
        metrics.record_query(result)
        if record_session:
            get_session().record_turn(result)
        return result

    except TimeoutError_ as e:
        logger.error("Workflow timed out: %s", e)
        error_result = {"final_output": None, "error": str(e), "user_query": query}
        metrics.record_query(error_result)
        if record_session:
            get_session().record_turn(error_result)
        return error_result
    except Exception as e:
        logger.error("Workflow interrupted: %s", e)
        error_result = {"final_output": None, "error": str(e), "user_query": query}
        metrics.record_query(error_result)
        if record_session:
            get_session().record_turn(error_result)
        return error_result
    finally:
        if use_alarm:
            signal.alarm(0)  # type: ignore[attr-defined]
            if 'original_handler' in locals() and original_handler:
                signal.signal(signal.SIGALRM, original_handler)  # type: ignore[attr-defined]


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


def run_daemon(interval: float = 30.0) -> None:
    """Daemon mode: continuous self-healing agent with heartbeat."""
    import time as _time

    from core.agent_session import reset_session
    from core.metrics import metrics

    session = reset_session()
    logger.info("=" * 50)
    logger.info("Daemon mode (session=%s, heartbeat=%ss)", session.session_id, interval)
    logger.info("=" * 50)

    # Self-check queries to probe agent health
    _health_checks = [
        "Write a Python function that adds two numbers and returns the result.",
        "Explain what a recursive function is in one sentence.",
    ]

    cycle = 0
    try:
        while True:
            cycle += 1
            logger.info("[Cycle %d] Heartbeat: health=%.2f, anomalies=%d%%",
                        cycle, session.health_score(), int(session.anomaly_rate() * 100))

            # Run health check query
            probe = _health_checks[cycle % len(_health_checks)]
            result = run_single_query(probe, timeout=30.0, record_session=False)
            session.record_turn(result)

            # Auto-recovery: if health drops below threshold, take action
            threshold = cfg("ESCALATION_THRESHOLD", 3) / 10.0
            if session.health_score() < max(threshold, 0.3):
                logger.warning(
                    "Health score below threshold (%.2f), triggering recovery",
                    threshold,
                )
                # Recovery action: reset session state
                session.record_recovery_event("reset", "Health low, state reset")
                reset_session()
                logger.info("Recovery: session state reset")

            # Persist session periodically
            if cycle % 5 == 0:
                path = session.save()
                logger.debug("Session saved to %s", path)

            # Print summary every 10 cycles
            if cycle % 10 == 0:
                summary = session.summary()
                logger.info("Session summary: %s", json.dumps(summary, default=str))

            _time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Daemon stopped by user (cycles=%d)", cycle)
        session.save()
        metrics.save_report()
        logger.info("Session and metrics saved.")


def show_stats() -> None:
    """Display immune memory and system statistics."""
    from core.agent_session import get_session
    from core.config import get as cfg_get
    from core.memory import memory_db

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
    print("\n  Configuration:")
    print(f"    Provider       : {cfg_get('LLM_PROVIDER', 'openai')}")
    print(f"    Worker Model   : {cfg_get('MAIN_LLM_MODEL', 'gpt-4o')}")
    print(f"    Monitor Model  : {cfg_get('MONITOR_LLM_MODEL', 'gpt-4o-mini')}")
    print(f"    Sandbox Mode   : {cfg_get('SANDBOX_MODE', 'simulated')}")
    print(f"    Max Iterations : {cfg_get('MAX_ITERATIONS', 5)}")
    print(f"    Escalation Thr : {cfg_get('ESCALATION_THRESHOLD', 3)}")

    # Session stats
    try:
        sess = get_session()
        s = sess.summary()
        print("\n  Session Stats:")
        print(f"    ID            : {s['session_id']}")
        print(f"    Turns         : {s['total_turns']}")
        print(f"    Health Score  : {s['health_score']}")
        print(f"    Anomaly Rate  : {s['anomaly_rate']}")
        print(f"    Recoveries    : {s['total_recoveries']}")
        print(f"    Uptime        : {s['uptime_seconds']}s")
    except Exception as e:
        print(f"\n  Session Stats: error reading ({e})")

    print("\n  Project Info:")
    print(f"    Path   : {os.path.dirname(os.path.abspath(__file__))}")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"    Logs   : {os.path.join(project_dir, 'logs')}")
    print(f"    Memory : {os.path.join(project_dir, '.immune_db')}")
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
            "  %(prog)s -d                      Daemon mode (self-healing)\n"
            "  %(prog)s -d --heartbeat 10       Daemon with 10s heartbeat\n"
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
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Daemon mode: continuous self-healing agent",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds for daemon mode (default: 30)",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.daemon:
        run_daemon(interval=args.heartbeat)
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
