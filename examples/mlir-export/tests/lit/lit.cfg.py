# lit configuration for the ggml-dialect lowering tests.
#
# Paths come from the environment rather than a generated lit.site.cfg.py, so the suite is runnable
# by hand without a configure step:
#
#   TSI_LIT_TOOLS_DIR=$PWD/build/bin \
#   TSI_LIT_LLVM_TOOLS_DIR=~/repo/mlir-compiler/build/_deps/llvm-build/bin \
#   ~/repo/mlir-compiler/build/_deps/llvm-build/bin/llvm-lit -v examples/mlir-export/tests/lit
import os

import lit.formats

config.name = "tsi-ggml"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

tools_dir = os.environ.get("TSI_LIT_TOOLS_DIR", "")
llvm_tools_dir = os.environ.get("TSI_LIT_LLVM_TOOLS_DIR", "")

config.test_exec_root = os.environ.get(
    "TSI_LIT_EXEC_ROOT", os.path.join(tools_dir or config.test_source_root, "lit-output")
)

# FileCheck and `not` come from the mlir-compiler LLVM build; tsi-ggml-opt from ours.
config.environment["PATH"] = os.pathsep.join(
    [p for p in (llvm_tools_dir, tools_dir, os.environ.get("PATH", "")) if p]
)

config.substitutions.append(("%tsi-ggml-opt", os.path.join(tools_dir, "tsi-ggml-opt")))
