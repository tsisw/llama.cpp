"""Structural check: the IR the exporter emits today matches the committed golden IR.

This exists to make replacing the string emitter with the MLIR C++ API safe. The numeric suite in
test_mlir_export.py only proves the graphs it can run produce the right numbers; it says nothing
about whether a rewritten emitter produced *the same IR* for the ops whose numerics happen to be
insensitive to the difference. This does.

Both sides are normalized by parsing and re-printing with MLIR before comparison, so the things
that legitimately differ between a hand-written string and MLIR's own printer all cancel:

  - affine maps inline (`affine_map<...>`) vs hoisted to module-level `#map` aliases
  - SSA numbering and auto-generated names (`%3` vs `%transposed`)
  - whitespace, operand-list line breaks, attribute ordering

What survives normalization is real structure: op sequence, types, shapes, permutations,
reassociation indices, iterator types, and constant values. If those differ, this fails.

Regenerating the goldens is deliberate, not automatic:

    ./build/bin/mlir-export-cases --emit-all /tmp/c
    for d in /tmp/c/*/; do cp "$d/forward.mlir" examples/mlir-export/tests/golden/"$(basename $d)".mlir; done

Only do that when you intend the IR to change, and review the diff.
"""
from pathlib import Path

import pytest

from tsi_mlir.ir import Context, Module

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _canonical(text, what):
    """Parse then print, so formatting and naming differences do not register as changes."""
    with Context():
        try:
            return Module.parse(text).operation.get_asm()
        except Exception as e:  # noqa: BLE001 - surface the parse failure with its source
            raise AssertionError(f"{what} is not parseable MLIR: {e}") from e


def test_emitted_ir_matches_golden(case):
    case_dir, meta = case
    name = meta["name"]

    golden_path = GOLDEN_DIR / f"{name}.mlir"
    if not golden_path.exists():
        pytest.fail(
            f"no golden IR for case {name!r} at {golden_path}. A new case must have its golden "
            f"committed, otherwise the case is unprotected against an emitter rewrite."
        )

    emitted = (case_dir / "forward.mlir").read_text()
    golden = golden_path.read_text()

    if not golden.strip():
        # An 'unsupported' case emits nothing; the golden records that, so emitted must be empty too.
        assert not emitted.strip(), (
            f"case {name!r} has an empty golden (recorded as unsupported) but now emits IR. "
            f"If the exporter gained support for it, regenerate the golden."
        )
        return

    assert emitted.strip(), f"case {name!r} emitted no IR but the golden is non-empty"

    got = _canonical(emitted, f"emitted IR for {name!r}")
    want = _canonical(golden, f"golden IR for {name!r}")

    if got != want:
        import difflib
        diff = "\n".join(difflib.unified_diff(
            want.splitlines(), got.splitlines(),
            fromfile=f"golden/{name}.mlir", tofile=f"emitted/{name}", lineterm="", n=3))
        raise AssertionError(
            f"case {name!r}: emitted IR differs structurally from the golden.\n"
            f"Both sides were parsed and re-printed, so this is NOT a formatting difference.\n\n"
            f"{diff}"
        )
