import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ViolationSeverity(Enum):
    CRITICAL = "critical"  # Must block — potential for real harm
    HIGH = "high"          # Must block — violates sandbox intent
    MEDIUM = "medium"      # Should block — suspicious but maybe safe
    LOW = "low"            # Warn — probably safe but unusual


@dataclass
class Violation:
    severity: ViolationSeverity
    rule: str
    message: str
    line: int = 0
    col: int = 0


@dataclass
class GuardResult:
    allowed: bool
    violations: list[Violation] = field(default_factory=list)
    sanitized_code: Optional[str] = None  # If we can auto-fix

    def add(self, severity: ViolationSeverity, rule: str, message: str,
            line: int = 0, col: int = 0):
        self.violations.append(Violation(severity, rule, message, line, col))
        if severity in (ViolationSeverity.CRITICAL, ViolationSeverity.HIGH):
            self.allowed = False


# ---------------------------------------------------------------------------
# Rule implementations — each is an AST visitor that checks one category
# ---------------------------------------------------------------------------

class _ImportChecker(ast.NodeVisitor):
    """Block all imports except explicitly allowed ones."""

    # Modules that are safe to import
    ALLOWED_MODULES = {
        # Math/stats
        "math", "statistics", "cmath",
        # Data structures
        "collections", "itertools", "functools", "operator",
        # DateTime
        "datetime", "dateutil",
        # String/text
        "string", "re", "textwrap",
        # Type hints
        "typing",
        # Visualization — LLM generates chart code via sandbox
        "plotly",
    }

    # Modules that are NEVER allowed under any circumstances
    BLOCKED_MODULES = {
        # System access
        "os", "sys", "subprocess", "shutil", "pathlib",
        "ctypes", "multiprocessing", "threading", "asyncio",
        # Network
        "socket", "http", "urllib", "requests", "aiohttp",
        "ftplib", "smtplib", "telnetlib",
        # Code execution
        "exec", "eval", "compile", "code", "codeop",
        "importlib", "__import__",
        # File system
        "io", "tempfile", "glob", "fnmatch",
        # Dangerous stdlib
        "pickle", "shelve", "marshal", "csv", "json",
        "webbrowser", "antigravity",
        # Process/system
        "signal", "resource", "pwd", "grp", "posix",
        "nt", "winreg",
        # GUI
        "tkinter", "wx", "pygame",
        # Database
        "sqlite3", "pymysql", "psycopg2",
        # Crypto (can be used for malicious purposes)
        "hashlib", "hmac", "secrets",
    }

    def __init__(self, result: GuardResult):
        self.result = result
        self._has_violation = False

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            module_name = alias.name.split(".")[0]  # Get base module
            self._check_module(module_name, node.lineno, node.col_offset)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            module_name = node.module.split(".")[0]
            self._check_module(module_name, node.lineno, node.col_offset)
            # Also check what's being imported
            for alias in node.names:
                self._check_from_import(node.module, alias.name, node.lineno)
        self.generic_visit(node)

    def _check_module(self, module: str, line: int, col: int):
        # Check blocked list first
        if module in self.BLOCKED_MODULES:
            self.result.add(
                ViolationSeverity.CRITICAL,
                "blocked_import",
                f"Import of '{module}' is explicitly blocked",
                line, col
            )
            return

        # Check if it's in allowed list
        if module not in self.ALLOWED_MODULES:
            self.result.add(
                ViolationSeverity.HIGH,
                "unrecognized_import",
                f"Import of '{module}' is not in the allowlist. "
                f"Allowed: {sorted(self.ALLOWED_MODULES)}",
                line, col
            )

    def _check_from_import(self, module: str, name: str, line: int):
        # Block dangerous from-imports even from "safe" modules
        dangerous_names = {
            "exec", "eval", "compile", "__import__",
            "open", "input", "breakpoint",
            "exit", "quit", "sys", "os",
        }
        if name in dangerous_names:
            self.result.add(
                ViolationSeverity.CRITICAL,
                "dangerous_from_import",
                f"Cannot import '{name}' from '{module}'",
                line
            )


class _DangerousCallChecker(ast.NodeVisitor):
    """Block calls to dangerous built-in functions."""

    BLOCKED_BUILTINS = {
        # Code execution
        "exec", "eval", "compile",
        # File I/O
        "open", "input",
        # Process control
        "exit", "quit", "breakpoint",
        # Attribute access tricks
        "getattr", "setattr", "delattr", "hasattr",
        # Type manipulation (can bypass restrictions)
        "type", "object", "__class__", "__base__", "__subclasses__",
        # Memory inspection
        "id", "vars", "dir", "globals", "locals",
    }

    def __init__(self, result: GuardResult):
        self.result = result

    def visit_Call(self, node: ast.Call):
        func_name = self._get_func_name(node)

        if func_name in self.BLOCKED_BUILTINS:
            self.result.add(
                ViolationSeverity.CRITICAL,
                "blocked_builtin_call",
                f"Call to '{func_name}'() is not allowed",
                node.lineno, node.col_offset
            )

        # Check for dangerous method calls like obj.__class__
        if isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            if attr_name.startswith("__") and attr_name.endswith("__"):
                self.result.add(
                    ViolationSeverity.HIGH,
                    "dunder_method_call",
                    f"Call to dunder method '{attr_name}' is restricted",
                    node.lineno, node.col_offset
                )

        self.generic_visit(node)

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


class _AttributeAccessChecker(ast.NodeVisitor):
    """Block dangerous attribute access patterns."""

    # Dunder attributes that can be used to escape sandbox
    BLOCKED_DUNDERS = {
        "__class__", "__base__", "__bases__", "__subclasses__",
        "__mro__", "__init__", "__globals__", "__code__",
        "__dict__", "__doc__", "__module__", "__import__",
        "__builtins__", "__name__", "__file__", "__path__",
    }

    def __init__(self, result: GuardResult):
        self.result = result

    def visit_Attribute(self, node: ast.Attribute):
        attr = node.attr

        # Block dunder access
        if attr in self.BLOCKED_DUNDERS:
            self.result.add(
                ViolationSeverity.HIGH,
                "dunder_access",
                f"Access to '{attr}' is restricted",
                node.lineno, node.col_offset
            )

        # Block common escape patterns
        if attr in ("globals", "locals", "vars"):
            self.result.add(
                ViolationSeverity.HIGH,
                "scope_access",
                f"Access to '{attr}' is restricted",
                node.lineno, node.col_offset
            )

        self.generic_visit(node)


class _StringPatternChecker(ast.NodeVisitor):
    """Check for dangerous patterns in string literals."""

    # Patterns that suggest escape attempts
    ESCAPE_PATTERNS = [
        (r"__\w+__", "dunder string pattern"),
        (r"\\x[0-9a-fA-F]{2}", "hex escape sequence"),
        (r"\\u[0-9a-fA-F]{4}", "unicode escape sequence"),
        (r"\\U[0-9a-fA-F]{8}", "long unicode escape"),
        (r"base64", "base64 encoding reference"),
        (r"pickle", "pickle serialization reference"),
        (r"marshal", "marshal serialization reference"),
        (r"subprocess", "subprocess reference"),
        (r"/etc/passwd", "system file reference"),
        (r"/etc/shadow", "system file reference"),
        (r"C:\\\\Windows", "Windows system path"),
        (r"\\127\\.0\\.0\\.1", "localhost reference"),
        (r"socket\\.connect", "socket connection pattern"),
    ]

    def __init__(self, result: GuardResult):
        self.result = result

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            self._check_string(node.value, node.lineno)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        # f-strings — check the parts
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                self._check_string(value.value, node.lineno)
        self.generic_visit(node)

    def _check_string(self, s: str, line: int):
        for pattern, name in self.ESCAPE_PATTERNS:
            if re.search(pattern, s, re.IGNORECASE):
                self.result.add(
                    ViolationSeverity.MEDIUM,
                    "suspicious_string",
                    f"String contains suspicious pattern: {name}",
                    line
                )


class _ControlFlowChecker(ast.NodeVisitor):
    """Limit control flow to prevent infinite loops and excessive computation."""

    MAX_LOOP_DEPTH = 2  # No nested loops beyond this
    MAX_FUNCTION_DEPTH = 2  # No nested function definitions

    def __init__(self, result: GuardResult):
        self.result = result
        self._loop_depth = 0
        self._func_depth = 0

    def visit_For(self, node: ast.For):
        self._loop_depth += 1
        if self._loop_depth > self.MAX_LOOP_DEPTH:
            self.result.add(
                ViolationSeverity.MEDIUM,
                "nested_loop",
                f"Loop nesting depth ({self._loop_depth}) exceeds limit ({self.MAX_LOOP_DEPTH})",
                node.lineno
            )
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While):
        self._loop_depth += 1
        if self._loop_depth > self.MAX_LOOP_DEPTH:
            self.result.add(
                ViolationSeverity.MEDIUM,
                "nested_loop",
                f"Loop nesting depth ({self._loop_depth}) exceeds limit ({self.MAX_LOOP_DEPTH})",
                node.lineno
            )
        # Warn about while True
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.result.add(
                ViolationSeverity.HIGH,
                "infinite_loop_risk",
                "'while True' detected — likely to hit timeout",
                node.lineno
            )
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._func_depth += 1
        if self._func_depth > self.MAX_FUNCTION_DEPTH:
            self.result.add(
                ViolationSeverity.MEDIUM,
                "nested_function",
                f"Function definition depth ({self._func_depth}) exceeds limit ({self.MAX_FUNCTION_DEPTH})",
                node.lineno
            )
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.result.add(
            ViolationSeverity.HIGH,
            "async_function",
            "Async functions are not allowed in sandbox",
            node.lineno
        )


class _ComprehensionChecker(ast.NodeVisitor):
    """Limit comprehensions to prevent memory exhaustion."""

    MAX_COMPREHENSIONS = 5  # Total comprehensions in code
    MAX_NESTED_COMPREHENSION = 1  # No nested comprehensions

    def __init__(self, result: GuardResult):
        self.result = result
        self._count = 0
        self._depth = 0

    def visit_ListComp(self, node: ast.ListComp):
        self._check_comp(node)
    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp):
        self._check_comp(node)

    def _check_comp(self, node):
        self._count += 1
        if self._count > self.MAX_COMPREHENSIONS:
            self.result.add(
                ViolationSeverity.MEDIUM,
                "too_many_comprehensions",
                f"Number of comprehensions ({self._count}) exceeds limit ({self.MAX_COMPREHENSIONS})",
                node.lineno
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Size and structure limits
# ---------------------------------------------------------------------------

MAX_CODE_LENGTH = 2000  # characters
MAX_LINES = 50
MAX_AST_NODES = 200


def _check_size(code: str, result: GuardResult):
    """Check code size limits."""
    if len(code) > MAX_CODE_LENGTH:
        result.add(
            ViolationSeverity.HIGH,
            "code_too_long",
            f"Code length ({len(code)}) exceeds limit ({MAX_CODE_LENGTH} chars)",
        )

    lines = code.split("\n")
    if len(lines) > MAX_LINES:
        result.add(
            ViolationSeverity.HIGH,
            "too_many_lines",
            f"Line count ({len(lines)}) exceeds limit ({MAX_LINES})",
        )

    try:
        tree = ast.parse(code)
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > MAX_AST_NODES:
            result.add(
                ViolationSeverity.MEDIUM,
                "too_complex",
                f"AST node count ({node_count}) exceeds limit ({MAX_AST_NODES})",
            )
    except SyntaxError as e:
        result.add(
            ViolationSeverity.HIGH,
            "syntax_error",
            f"Code has syntax error: {e.msg} at line {e.lineno}",
            e.lineno or 0,
            e.offset or 0,
        )


# ---------------------------------------------------------------------------
# Main guard function
# ---------------------------------------------------------------------------

def validate_sandbox_code(code: str) -> GuardResult:
    """
    Validate code for sandbox execution.

    Returns GuardResult with:
    - allowed: True if code passes all checks
    - violations: List of any issues found

    This is a FAST check — no code execution, just AST analysis.
    """
    result = GuardResult(allowed=True)

    # 1. Size limits
    _check_size(code, result)

    # 2. Parse AST (if not already failed)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result  # Already recorded in _check_size

    # 3. Run all checkers
    checkers = [
        _ImportChecker(result),
        _DangerousCallChecker(result),
        _AttributeAccessChecker(result),
        _StringPatternChecker(result),
        _ControlFlowChecker(result),
        _ComprehensionChecker(result),
    ]

    for checker in checkers:
        checker.visit(tree)

    return result


def get_guard_summary(result: GuardResult) -> str:
    """Format guard results for logging / LLM feedback."""
    if result.allowed:
        return "Code passed all safety checks."

    lines = ["Code failed safety checks:"]
    for v in result.violations:
        loc = f" (line {v.line})" if v.line else ""
        lines.append(f"  [{v.severity.value.upper()}] {v.rule}{loc}: {v.message}")

    return "\n".join(lines)