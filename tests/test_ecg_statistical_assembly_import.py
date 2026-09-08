from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_statistical_assembler_imports_without_plotting_or_torch_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
import runpy

real_import = builtins.__import__

def import_without_heavy_runtime(name, globals=None, locals=None, fromlist=(), level=0):
    if (
        name == "torch"
        or name.startswith("torch.")
        or name == "matplotlib"
        or name.startswith("matplotlib.")
    ):
        raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_heavy_runtime
runpy.run_path("scripts/assemble_open_ecg_statistical_addendum.py", run_name="trust_ecg_assembly_import_test")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
