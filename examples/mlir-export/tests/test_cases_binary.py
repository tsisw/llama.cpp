"""Stage 1 alone: the generator must produce a well-formed, parseable case dir."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES_BIN = Path(__file__).resolve().parents[3] / "build" / "bin" / "test-mlir-export-cases"


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_list_includes_add():
    out = subprocess.run([str(CASES_BIN), "--list"], capture_output=True, text=True, check=True)
    assert "add" in out.stdout.split()


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_emit_add_writes_wellformed_case(tmp_path):
    subprocess.run([str(CASES_BIN), "--emit", "add", str(tmp_path)], check=True)

    meta = json.loads((tmp_path / "case.json").read_text())
    assert meta["name"] == "add"
    assert meta["expect"] == "pass"
    assert len(meta["args"]) == 2

    mlir = (tmp_path / "forward.mlir").read_text()
    assert mlir.lstrip().startswith("module {")
    assert "func.func @forward" in mlir
    assert "llvm.emit_c_interface" in mlir

    # Reference must be the real elementwise sum of the two inputs we wrote out.
    a = np.fromfile(tmp_path / "input_0.bin", dtype=np.float32)
    b = np.fromfile(tmp_path / "input_1.bin", dtype=np.float32)
    exp = np.fromfile(tmp_path / "expected_0.bin", dtype=np.float32)
    n = int(np.prod(meta["output"]["shape"]))
    assert exp.size == n
    np.testing.assert_array_equal(exp, a + b)


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_emit_is_deterministic(tmp_path):
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    for d in (d1, d2):
        d.mkdir()
        subprocess.run([str(CASES_BIN), "--emit", "add", str(d)], check=True)
    for f in ("input_0.bin", "input_1.bin", "expected_0.bin", "forward.mlir"):
        assert (d1 / f).read_bytes() == (d2 / f).read_bytes(), f"{f} not deterministic"
