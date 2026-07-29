// An mlir-opt-style driver for the ggml dialect, so the lowering can be tested per op with
// FileCheck instead of only end to end against golden IR:
//
//   tsi-ggml-opt --convert-ggml-to-linalg input.mlir
//
// The end-to-end suite proves the whole pipeline agrees with ggml numerically, but it cannot say
// which linalg a single ggml op lowers to without running a whole graph through the TSI compiler.
// This can, from a hand-written .mlir file, in milliseconds.
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "tsi/export/Exporter.h"   // mlir_export_error

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

namespace {

// Wraps convertGgmlToLinalg as a real pass so it is reachable by name from the command line and
// composes with anything else mlir-opt offers.
struct ConvertGgmlToLinalgPass
    : public PassWrapper<ConvertGgmlToLinalgPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGgmlToLinalgPass)

    llvm::StringRef getArgument() const final { return "convert-ggml-to-linalg"; }
    llvm::StringRef getDescription() const final { return "Lower the ggml dialect to linalg"; }

    void getDependentDialects(DialectRegistry & registry) const override {
        registry.insert<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect,
                        math::MathDialect, func::FuncDialect>();
    }

    void runOnOperation() final {
        try {
            tsi::mlir_export::convertGgmlToLinalg(getOperation());
        } catch (const tsi::mlir_export::mlir_export_error &) {
            // The lowering reports the reason through MLIR diagnostics before throwing; turn the
            // exception into a pass failure so mlir-opt exits non-zero as expected.
            signalPassFailure();
        }
    }
};

}  // namespace

int main(int argc, char ** argv) {
    PassRegistration<ConvertGgmlToLinalgPass>();

    DialectRegistry registry;
    registry.insert<tsi::ggml::GgmlDialect, func::FuncDialect, arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, math::MathDialect>();

    return asMainReturnCode(MlirOptMain(argc, argv, "TSI ggml dialect optimizer driver\n", registry));
}
