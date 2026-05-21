#!/usr/bin/env python3
"""
Project setup and configuration helper.

Automates:
  - .env file creation from .env.example
  - Dependency installation
  - Optional dependency group selection
  - Directory structure verification
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def print_banner():
    print("=" * 55)
    print("  Symbiotic Cognitive Immune System Agent")
    print("  Setup & Configuration Helper")
    print("=" * 55)


def step(msg: str):
    print(f"\n  [*] {msg}...")


def check_env():
    """Check if .env exists, create from example if not."""
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"

    if env_path.exists():
        print(f"  [OK] .env already exists")
        return

    if not example_path.exists():
        print(f"  [!!] .env.example not found!")
        return

    print("  [..] .env not found. Creating from .env.example...")
    content = example_path.read_text()
    env_path.write_text(content)
    print(f"  [OK] Created .env — edit it to add your API keys")


def check_dirs():
    """Ensure required directories exist."""
    for d in ["logs", "escalations", "benchmarks"]:
        (ROOT / d).mkdir(exist_ok=True)
    print(f"  [OK] Directories ready")


def get_optional_groups() -> list[str]:
    """Ask user which optional dependency groups to install."""
    print("\n  Optional dependency groups:")
    print("    [1] memory  - ChromaDB persistent immune memory")
    print("    [2] webui   - Streamlit web interface")
    print("    [3] api     - FastAPI REST API server")
    print("    [4] dev     - Testing & linting tools")
    print("    [5] all     - Everything above")
    print("    [0] none    - Core only (default)")

    choice = input("  Select [0-5] (default: 0): ").strip() or "0"

    mapping = {
        "1": ["memory"],
        "2": ["webui"],
        "3": ["api"],
        "4": ["dev"],
        "5": ["memory", "webui", "api", "dev"],
        "0": [],
    }
    return mapping.get(choice, [])


def install_deps(groups: list[str]):
    """Install core + selected optional dependencies."""
    step("Installing core dependencies")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        capture_output=True,
    )

    if groups:
        extras = ",".join(groups)
        step(f"Installing optional: {extras}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f".[{extras}]"],
            cwd=str(ROOT),
            capture_output=True,
        )
    print(f"  [OK] Dependencies installed")


def run_health_check():
    """Verify core modules import correctly."""
    step("Running import health check")
    modules = [
        "core.state", "core.logger", "core.config",
        "core.memory", "core.sandbox", "core.escalation",
        "core.metrics", "core.viz",
    ]
    failures = 0
    for mod in modules:
        try:
            __import__(mod)
            print(f"    [OK] {mod}")
        except ImportError as e:
            print(f"    [!!] {mod}: {e}")
            failures += 1

    if failures == 0:
        print(f"  [OK] All core modules import successfully")


def main():
    print_banner()

    check_dirs()
    check_env()

    groups = get_optional_groups()
    install_deps(groups)

    run_health_check()

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print()
    print("  Next steps:")
    print("    1. Edit .env with your API keys")
    print("    2. Run: python immune_agent.py --stats")
    print("    3. Run: python -m pytest tests/ -v")
    print("    4. Try: streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
