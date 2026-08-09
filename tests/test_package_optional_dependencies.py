"""Regression tests for optional dependency isolation at the package root."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_core_ecg_import_does_not_require_matplotlib() -> None:
    code = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise RuntimeError("core TRUST-ECG import attempted to import matplotlib")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import trust_icu.ecg_data  # noqa: F401
        print("core_ecg_import_ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "core_ecg_import_ok"
