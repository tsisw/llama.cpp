"""Teach the TSI compiler driver to accept an already-lowered linalg MLIR module.

TXEBackend subclasses normally implement convert_to_linalg to turn a torch/ONNX model into linalg.
The ggml exporter (exporter.h) already emits linalg MLIR text, so the "conversion" is just a parse.

Shared by compile_graph_fpga.py (AOT, for FPGA/posix bundles) and tests/test_mlir_export.py (JIT).
"""
from tsi_mlir.ir import Module
from tsavorite.txe_backend.txe_backend import TXEBackend


class RawGraphBackend(TXEBackend):
    """The model is already whole-graph linalg MLIR text; parse it into the txe context."""

    def convert_to_linalg(self, model, input_types, *, func_name=None, log_dir=None,
                          verbose=False, **kwargs):
        return Module.parse(model)
