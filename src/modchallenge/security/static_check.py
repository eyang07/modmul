"""AST-based static analysis for submitted code.

The scanner is intentionally narrow: it catches obvious, mechanical violations
of the prohibited-practices rules. Subtler attacks (chunked multiplication,
CRT recombination spread across helper functions, computed answers in load(),
etc.) are out of scope and must be caught by manual review.

Layered with the sandbox package allowlist (which prevents `import sympy` from
even succeeding at runtime), this scanner provides a fast pre-load check that
gives contestants an immediate, actionable error rather than a runtime
ImportError partway through evaluation.

Rules enforced:
- forbidden-import: importing a banned package (sympy, gmpy2, mpmath, flint,
  decimal, networking libs, subprocess libs, ctypes)
- dynamic-exec: calling eval / exec / compile / __import__
- modmul-shortcut: the patterns `int(a) * int(b) % int(p)`,
  `(int(a) * int(b)) % int(p)`, `pow(int(a), int(b), int(p))`,
  and arithmetic equivalents that compute the modular product directly
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Top-level package names that may not be imported. We match on the root
# package; e.g. banning "urllib" also bans "urllib.request".
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        # Symbolic / arbitrary-precision math (must not be used to compute
        # the modular product; sandbox additionally won't install them).
        "sympy",
        "gmpy2",
        "mpmath",
        "flint",
        "decimal",
        # Network access of any kind.
        "socket",
        "ssl",
        "urllib",
        "urllib2",
        "urllib3",
        "requests",
        "httpx",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "poplib",
        "imaplib",
        "xmlrpc",
        "asyncio",
        "aiohttp",
        # Subprocess / system command execution.
        "subprocess",
        "multiprocessing",
        # Low-level FFI used to bypass Python-level checks.
        "ctypes",
        "cffi",
    }
)

# Functions that perform dynamic code execution. Any direct call (not just
# any reference) to one of these names is flagged.
DYNAMIC_EXEC_NAMES: frozenset[str] = frozenset(
    {"eval", "exec", "compile", "__import__"}
)

# os.* attribute names that spawn processes / execute commands.
OS_PROCESS_ATTRS: frozenset[str] = frozenset(
    {
        "system",
        "popen",
        "exec",
        "execv",
        "execvp",
        "execvpe",
        "execve",
        "execl",
        "execlp",
        "execle",
        "spawn",
        "spawnv",
        "spawnvp",
        "spawnvpe",
        "spawnve",
        "spawnl",
        "spawnlp",
        "spawnle",
        "fork",
        "forkpty",
    }
)


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """A single rule violation reported by the scanner."""

    file: str
    line: int
    col: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.file}:{self.line}:{self.col}  [{self.rule}] {self.message}"


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

def _is_int_call(node: ast.AST) -> bool:
    """Return True if `node` is a Call to the built-in `int(...)`."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "int"
    )


def _is_modmul_pattern(node: ast.BinOp) -> bool:
    """Detect `int(_) * int(_) % int(_)` and equivalents.

    Matches:
      `(int(a) * int(b)) % int(p)`
      `int(a) * int(b) % int(p)` (parses identically — Python operator
      precedence makes `*` bind tighter than `%`).
    """
    if not isinstance(node.op, ast.Mod):
        return False
    if not _is_int_call(node.right):
        return False
    inner = node.left
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult)):
        return False
    return _is_int_call(inner.left) and _is_int_call(inner.right)


def _is_modpow_pattern(node: ast.Call) -> bool:
    """Detect `pow(int(_), int(_), int(_))` (3-arg modular exponentiation)."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "pow"):
        return False
    if len(node.args) != 3:
        return False
    return all(_is_int_call(arg) for arg in node.args)


def _root_package(name: str) -> str:
    """Given a dotted module path, return the top-level package."""
    return name.split(".", 1)[0]


class _Scanner(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file = file_path
        self.findings: list[Finding] = []

    # -- helpers --------------------------------------------------------

    def _add(self, node: ast.AST, rule: str, message: str) -> None:
        self.findings.append(
            Finding(
                file=self.file,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                rule=rule,
                message=message,
            )
        )

    # -- imports --------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = _root_package(alias.name)
            if root in FORBIDDEN_IMPORTS:
                self._add(
                    node,
                    "forbidden-import",
                    f"import of '{alias.name}' is not allowed",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root = _root_package(node.module)
            if root in FORBIDDEN_IMPORTS:
                self._add(
                    node,
                    "forbidden-import",
                    f"import from '{node.module}' is not allowed",
                )
        self.generic_visit(node)

    # -- calls ----------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        # Dynamic execution: eval/exec/compile/__import__
        if isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_EXEC_NAMES:
            self._add(
                node,
                "dynamic-exec",
                f"call to '{node.func.id}' is not allowed",
            )
        # os.system / os.popen / os.execX / os.spawnX / os.fork
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr in OS_PROCESS_ATTRS
        ):
            self._add(
                node,
                "subprocess",
                f"call to 'os.{node.func.attr}' is not allowed",
            )
        # pow(int(a), int(b), int(p)) — modular exponentiation shortcut
        if _is_modpow_pattern(node):
            self._add(
                node,
                "modmul-shortcut",
                "pow(int(_), int(_), int(_)) computes the modular product "
                "directly and is not allowed",
            )
        self.generic_visit(node)

    # -- binary ops -----------------------------------------------------

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if _is_modmul_pattern(node):
            self._add(
                node,
                "modmul-shortcut",
                "int(_) * int(_) % int(_) computes the modular product "
                "directly and is not allowed",
            )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_source(source: str, file_path: str = "<string>") -> list[Finding]:
    """Scan a string of Python source and return findings."""
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return [
            Finding(
                file=file_path,
                line=exc.lineno or 0,
                col=exc.offset or 0,
                rule="syntax-error",
                message=f"could not parse source: {exc.msg}",
            )
        ]
    scanner = _Scanner(file_path)
    scanner.visit(tree)
    return scanner.findings


def check_file(path: Path) -> list[Finding]:
    """Scan a single .py file and return findings."""
    source = path.read_text(encoding="utf-8")
    return check_source(source, str(path))


def check_submission(submission_dir: Path) -> list[Finding]:
    """Scan every .py file under `submission_dir` and aggregate findings.

    Files are scanned in sorted order for deterministic output.
    """
    submission_dir = Path(submission_dir)
    findings: list[Finding] = []
    for py_file in sorted(submission_dir.rglob("*.py")):
        findings.extend(check_file(py_file))
    return findings
