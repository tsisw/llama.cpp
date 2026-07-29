"""End-to-end: a ggml graph exported to linalg MLIR compiles and executes with numerics matching
ggml's own CPU result.

Each case directory is produced by mlir-export-cases. case.json's "expect" field decides the
assertion: pass -> must match, unsupported -> xfail with the exporter's reason, mismatch -> must
NOT match (that case exists to prove this comparison is not vacuous).
"""
import numpy as np
import pytest
import torch

from tsavorite.compiler_config import TXECompilerConfig
from tsi_raw_backend import RawGraphBackend


def _config(target):
    if target == "ten":
        return TXECompilerConfig.for_ten(log_mlir=True, enable_tvu=True, enable_tmu=True)
    return TXECompilerConfig(log_mlir=True)


def _load(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape)


def test_case_matches_ggml_reference(case, target, tmp_path):
    case_dir, meta = case

    if meta["expect"] == "unsupported":
        pytest.xfail(f"exporter does not support case {meta['name']!r} "
                     f"(see stderr from mlir-export-cases)")

    mlir = (case_dir / "forward.mlir").read_text()
    assert "func.func @forward" in mlir, "case emitted no forward function"

    inputs = [torch.from_numpy(_load(case_dir / a["file"], a["shape"]).copy())
              for a in meta["args"]]
    expected = _load(case_dir / meta["output"]["file"], meta["output"]["shape"])

    runner = RawGraphBackend(_config(target)).compile(
        model=mlir, input_types=[], compilation_type="jit",
        output_dir=str(tmp_path / "compiled"), verbose=False,
    )

    runner.runtime_init()
    try:
        got = runner.forward(*inputs)
    finally:
        runner.runtime_finalize()

    got = np.asarray(got.detach() if hasattr(got, "detach") else got, dtype=np.float32)
    got = got.reshape(expected.shape)

    close = np.allclose(got, expected, rtol=meta["rtol"], atol=meta["atol"])

    if meta["expect"] == "mismatch":
        assert not close, (
            f"case {meta['name']!r} carries a deliberately corrupted reference but compared EQUAL. "
            f"The comparison is not actually checking anything."
        )
        return

    if not close:
        diff = np.abs(got - expected)
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        denom = np.maximum(np.abs(expected), 1e-30)
        raise AssertionError(
            f"case {meta['name']!r} mismatch: max abs err {diff.max():.3e} at {idx} "
            f"(got {got[idx]:.8g} vs expected {expected[idx]:.8g}), "
            f"max rel err {np.max(diff / denom):.3e}, "
            f"tolerance rtol={meta['rtol']} atol={meta['atol']}"
        )
