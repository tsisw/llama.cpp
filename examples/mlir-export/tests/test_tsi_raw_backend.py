"""The extraction must leave both consumers importable and behaviorally identical."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_raw_backend_importable_and_overrides_convert_to_linalg():
    from tsi_raw_backend import RawGraphBackend
    from tsavorite.txe_backend.txe_backend import TXEBackend

    assert issubclass(RawGraphBackend, TXEBackend)
    # The whole point of the class: convert_to_linalg is overridden, not inherited.
    assert RawGraphBackend.convert_to_linalg is not TXEBackend.convert_to_linalg


def test_raw_backend_parses_linalg_text():
    # Module.parse needs an ambient MLIR Context. In the real path TXEBackend.compile establishes
    # one before calling convert_to_linalg, so a unit test has to supply its own.
    from tsi_mlir.ir import Context

    from tsi_raw_backend import RawGraphBackend

    mlir = """
    module {
      func.func @forward(%arg0: tensor<4xf32> {txe.name = "input_0"})
          -> (tensor<4xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
        return %arg0 : tensor<4xf32>
      }
    }
    """
    with Context():
        mod = RawGraphBackend.convert_to_linalg(None, mlir, [])
    assert "func.func @forward" in str(mod)


def test_compile_graph_fpga_still_imports():
    # The AOT driver must keep working after the extraction.
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "compile_graph_fpga.py"
    spec = importlib.util.spec_from_file_location("compile_graph_fpga", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "RawGraphBackend")
    assert hasattr(mod, "main")
