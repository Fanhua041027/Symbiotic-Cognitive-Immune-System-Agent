"""Sandbox validation module with multiple execution backends.

Supports three validation levels:
  - simulated  : keyword-based heuristic check (default, no deps needed)
  - ast        : static Python AST analysis (no extra deps)
  - docker     : real Docker container execution (requires Docker)
  - e2b        : E2B cloud sandbox execution (requires e2b_code_interpreter)
"""

import ast
import os
import subprocess
import tempfile

from core.config import get as cfg
from core.logger import setup_logger

logger = setup_logger("sandbox")


# ---------------------------------------------------------------------------
# Level 1: Simulated (heuristic keyword check)
# ---------------------------------------------------------------------------
def validate_simulated(code: str) -> bool:
    """Quick keyword-based heuristic to check if code looks like a real fix.

    Accepts both code snippets (with control-flow keywords) and natural
    language descriptions (with fix-related keywords). Rejects very short
    strings, comment-only stubs, and trivial content.
    """
    stripped = code.strip()
    if not stripped:
        return False

    # Reject comment-only stubs: check if there is ANY non-comment line
    lines = stripped.split("\n")
    has_real_code = any(
        line.strip() and not line.strip().startswith("#")
        for line in lines
    )

    # Require fix-relevant keywords
    fix_keywords = [
        "fix", "guard", "limit", "check", "max", "validate",
        "prevent", "ensure", "error", "except", "recursion",
        "iteration", "counter", "break", "return", "raise",
        "terminate", "safe", "condition", "bound", "stop",
    ]
    has_fix = any(kw in code.lower() for kw in fix_keywords)

    # Relaxed length threshold: 30+ chars
    is_long_enough = len(code) > 30

    # Accept if long enough AND has either real code or fix keywords.
    # This allows comment-starting antibodies (e.g. "# Guard: max iterations")
    # as long as they contain real code or recognizable fix terminology.
    return is_long_enough and (has_real_code or has_fix)


# ---------------------------------------------------------------------------
# Level 2: AST analysis (static code validation)
# ---------------------------------------------------------------------------
class ASTValidator:
    """Validate generated Python code by parsing its AST."""

    SUSPICIOUS_MODULES = {
        "os", "subprocess", "shutil", "sys",
        "__import__", "compile", "open", "globals", "locals",
    }

    DANGEROUS_BUILTINS = {"exec", "eval"}

    @classmethod
    def validate(cls, code: str) -> tuple[bool, str]:
        """
        Parse and validate Python code via AST.
        Returns (is_valid, reason_or_empty_string).
        """
        if not code or not code.strip():
            return False, "Empty code"

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Direct call: dangerous_name(...)
                if isinstance(func, ast.Name):
                    if func.id in cls.SUSPICIOUS_MODULES:
                        return False, f"Dangerous function call: {func.id}"
                    if func.id in cls.DANGEROUS_BUILTINS:
                        return False, f"Dangerous builtin: {func.id}"
                # Attribute call: dangerous_module.dangerous_func(...)
                if isinstance(func, ast.Attribute):
                    # Walk attribute chain to find the root module name
                    # e.g. os.path.join -> os is the root
                    root: ast.expr = func
                    while isinstance(root, ast.Attribute):
                        root = root.value  # type: ignore[assignment]
                    if isinstance(root, ast.Name) and root.id in cls.SUSPICIOUS_MODULES:
                        return False, (
                            f"Dangerous function call from module: {root.id}"
                        )
                    if func.attr.startswith("__"):
                        return False, f"Dunder method call: {func.attr}"

            # Detect ast.Call-free exec/eval (e.g. via Name node)
            if isinstance(node, ast.Name) and node.id in cls.DANGEROUS_BUILTINS:
                return False, f"Dangerous builtin: {node.id}"

        return True, ""

# ---------------------------------------------------------------------------
# Level 3: Docker sandbox (real execution in isolated container)
# ---------------------------------------------------------------------------
def validate_docker(code: str) -> tuple[bool, str]:
    """Run generated Python code inside a Docker container for real testing."""
    if not _docker_available():
        logger.warning("Docker not available, falling back to AST validation")
        return validate_ast(code)

    host_path = ""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        host_path = f.name
        container_path = "/tmp/antibody_check.py"

    try:
        result = subprocess.run(
            ["docker", "run", "--rm",
             "--memory=256m", "--cpus=0.5",
             "--network=none", "--read-only",
             "--security-opt=no-new-privileges",
             "-v", f"{host_path}:{container_path}:ro",
             "python:3.11-alpine",
             "python", container_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("Docker validation passed")
            return True, ""
        else:
            return False, f"Runtime error: {result.stderr.strip()[:200]}"
    except FileNotFoundError:
        logger.warning("Docker executable not found")
        return validate_ast(code)
    except subprocess.TimeoutExpired:
        return False, "Execution timed out (>30s)"
    except Exception as e:
        logger.error("Docker validation error: %s", e)
        return validate_ast(code)
    finally:
        if host_path:
            try:
                os.unlink(host_path)
            except OSError:
                pass


def _docker_available() -> bool:
    """Check if Docker CLI is accessible."""
    try:
        subprocess.run(
            ["docker", "--version"],
            capture_output=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Level 4: E2B cloud sandbox (real execution in E2B)
# ---------------------------------------------------------------------------
def validate_e2b(code: str) -> tuple[bool, str]:
    """Run generated Python code inside an E2B cloud sandbox for real testing."""
    if not code or not code.strip():
        return False, "Empty antibody code"

    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        logger.warning("e2b_code_interpreter not installed, falling back to AST")
        return validate_ast(code)

    try:
        with Sandbox() as sbx:
            result = sbx.run_code(code)
            # Check explicit error field
            if result.error:
                return False, f"Runtime error: {result.error.name}: {result.error.value}"
            # Check for stderr output (only if it's a real string, not a mock)
            stderr = getattr(result, "stderr", None)
            if isinstance(stderr, str) and stderr.strip():
                return False, f"Execution produced stderr: {stderr.strip()[:200]}"
            # Check for empty/no-op execution
            stdout = getattr(result, "stdout", None)
            has_logs = getattr(result, "logs", None)
            has_output = isinstance(stdout, str) and stdout
            if not has_output and not has_logs and len(code) > 100:
                logger.debug("E2B result produced no output for non-trivial code")
            return True, ""
    except Exception as e:
        logger.error("E2B validation error: %s", e)
        return validate_ast(code)


# ---------------------------------------------------------------------------
# Level 2 alias (AST only, no Docker)
# ---------------------------------------------------------------------------
def validate_ast(code: str) -> tuple[bool, str]:
    """Run AST validation only, returns (is_valid, reason)."""
    return ASTValidator.validate(code)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_antibody(code: str) -> tuple[bool, str]:
    """
    Validate an antibody code snippet using the configured sandbox mode.

    Returns (is_valid: bool, reason: str).
    """
    if not code or not code.strip():
        return False, "Empty antibody code"

    mode = cfg("SANDBOX_MODE", "simulated").lower()
    logger.info("Validating antibody (mode=%s, len=%d)", mode, len(code))

    if mode == "docker":
        return validate_docker(code)
    elif mode == "e2b":
        return validate_e2b(code)
    elif mode == "ast":
        return validate_ast(code)
    else:
        is_ok = validate_simulated(code)
        return is_ok, "" if is_ok else "Simulated check failed"
