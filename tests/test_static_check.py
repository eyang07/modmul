"""Tests for the static-analysis scanner."""

from __future__ import annotations

import textwrap

import pytest

from modchallenge.security.static_check import (
    Finding,
    check_source,
    check_submission,
)


def _rules(findings: list[Finding]) -> list[str]:
    return [f.rule for f in findings]


# ---------------------------------------------------------------------------
# Clean code (no findings)
# ---------------------------------------------------------------------------

def test_clean_model_passes() -> None:
    src = textwrap.dedent(
        """
        import torch
        import numpy as np
        from modchallenge.interface.base_model import ModularMultiplicationModel

        class MyModel(ModularMultiplicationModel):
            def load(self, model_dir):
                self.model = torch.load(model_dir + "/weights.pt")

            def preprocess_a(self, a):
                return [int(c) for c in a]

            def predict_digits(self, a_enc, b_enc, p_enc):
                output = self.model(a_enc, b_enc, p_enc)
                return list(output)
        """
    )
    assert check_source(src) == []


def test_representation_conversion_passes() -> None:
    """int() and modular arithmetic for representation conversion is fine."""
    src = textwrap.dedent(
        """
        def encode_padic(a_str, p_str, depth=10):
            a = int(a_str)
            p = int(p_str)
            digits = []
            for _ in range(depth):
                digits.append(a % p)
                a //= p
            return digits
        """
    )
    assert check_source(src) == []


def test_crt_setup_against_small_moduli_passes() -> None:
    """Setting up CRT moduli from p (without involving a or b) is fine."""
    src = textwrap.dedent(
        """
        def crt_residues(p_str, small_primes):
            p = int(p_str)
            return [p % q for q in small_primes]
        """
    )
    assert check_source(src) == []


# ---------------------------------------------------------------------------
# Forbidden imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "import_line",
    [
        "import sympy",
        "import gmpy2",
        "import mpmath",
        "import flint",
        "import decimal",
        "import socket",
        "import urllib",
        "import urllib.request",
        "import requests",
        "import httpx",
        "import http.client",
        "import ftplib",
        "import subprocess",
        "import multiprocessing",
        "import ctypes",
        "import cffi",
    ],
)
def test_forbidden_import_flagged(import_line: str) -> None:
    findings = check_source(import_line)
    assert "forbidden-import" in _rules(findings), (
        f"expected forbidden-import for {import_line!r}, got {findings}"
    )


@pytest.mark.parametrize(
    "from_line",
    [
        "from sympy import nextprime",
        "from gmpy2 import mpz",
        "from urllib.request import urlopen",
        "from subprocess import run",
        "from ctypes import c_int",
    ],
)
def test_forbidden_from_import_flagged(from_line: str) -> None:
    findings = check_source(from_line)
    assert "forbidden-import" in _rules(findings)


def test_safe_import_not_flagged() -> None:
    src = textwrap.dedent(
        """
        import torch
        import numpy as np
        import math
        import json
        from pathlib import Path
        """
    )
    assert check_source(src) == []


# ---------------------------------------------------------------------------
# Dynamic execution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "snippet",
    [
        "x = eval('1 + 2')",
        "exec('y = 3')",
        "code = compile('1+1', '<s>', 'eval')",
        "mod = __import__('os')",
    ],
)
def test_dynamic_exec_flagged(snippet: str) -> None:
    findings = check_source(snippet)
    assert "dynamic-exec" in _rules(findings)


def test_method_named_eval_not_flagged() -> None:
    """A user method or attribute happening to be named eval is not flagged."""
    src = textwrap.dedent(
        """
        class Foo:
            def eval(self, x):
                return x + 1

        f = Foo()
        f.eval(3)
        """
    )
    assert check_source(src) == []


# ---------------------------------------------------------------------------
# Subprocess / OS process spawning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "snippet",
    [
        "import os; os.system('echo hi')",
        "import os; os.popen('ls')",
        "import os; os.execv('/bin/ls', [])",
        "import os; os.fork()",
    ],
)
def test_os_process_calls_flagged(snippet: str) -> None:
    findings = check_source(snippet)
    rules = _rules(findings)
    assert "subprocess" in rules


# ---------------------------------------------------------------------------
# Modular multiplication shortcut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "snippet",
    [
        "answer = int(a) * int(b) % int(p)",
        "answer = (int(a) * int(b)) % int(p)",
        "answer = pow(int(a), int(b), int(p))",
    ],
)
def test_modmul_shortcut_flagged(snippet: str) -> None:
    findings = check_source(snippet)
    assert "modmul-shortcut" in _rules(findings), (
        f"expected modmul-shortcut for {snippet!r}, got {findings}"
    )


def test_pow_int_const_int_not_flagged() -> None:
    """`pow(int(a), 1, int(p))` is just int(a) % int(p) — a reduction, not the
    modular product. Don't false-flag it."""
    src = "x = pow(int(a), 1, int(p))"
    findings = check_source(src)
    assert "modmul-shortcut" not in _rules(findings)


def test_int_alone_not_flagged() -> None:
    """Bare int() conversions are not a shortcut."""
    src = textwrap.dedent(
        """
        a_int = int(a)
        b_int = int(b)
        digits_a = [(a_int >> i) & 1 for i in range(64)]
        """
    )
    assert check_source(src) == []


def test_int_times_int_without_mod_not_flagged() -> None:
    """Multiplication without the mod step isn't the shortcut."""
    src = "product = int(a) * int(b)"
    assert check_source(src) == []


def test_int_mod_int_without_mul_not_flagged() -> None:
    """Reducing a single value mod p (no multiplication) isn't the shortcut."""
    src = "reduced = int(a) % int(p)"
    assert check_source(src) == []


def test_pow_two_arg_not_flagged() -> None:
    """Two-arg pow (no modulus) is fine."""
    src = "x = pow(int(a), 2)"
    assert check_source(src) == []


# ---------------------------------------------------------------------------
# Submission-level scan
# ---------------------------------------------------------------------------

def test_check_submission_aggregates_files(tmp_path) -> None:
    (tmp_path / "model.py").write_text(
        "import sympy\n"
        "answer = int(a) * int(b) % int(p)\n"
    )
    (tmp_path / "helper.py").write_text(
        "import socket\n"
    )
    (tmp_path / "clean.py").write_text(
        "import torch\n"
    )

    findings = check_submission(tmp_path)
    rules = _rules(findings)
    assert rules.count("forbidden-import") == 2  # sympy + socket
    assert rules.count("modmul-shortcut") == 1


def test_check_submission_skips_non_python(tmp_path) -> None:
    (tmp_path / "weights.bin").write_bytes(b"\x00\x01")
    (tmp_path / "manifest.json").write_text('{"entry_class": "model.X"}')
    (tmp_path / "model.py").write_text("import torch")
    findings = check_submission(tmp_path)
    assert findings == []


def test_check_submission_handles_subdirectories(tmp_path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "inner.py").write_text("import sympy")
    (tmp_path / "model.py").write_text("import torch")
    findings = check_submission(tmp_path)
    assert _rules(findings) == ["forbidden-import"]


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------

def test_syntax_error_reported_not_crash() -> None:
    findings = check_source("def broken(:")
    assert _rules(findings) == ["syntax-error"]
