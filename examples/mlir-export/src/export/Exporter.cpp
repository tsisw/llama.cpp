// The public entry point: import the ggml graph into the ggml dialect, lower it to linalg, verify,
// print.
//
// Splitting import from lowering is the point of the dialect: "did we read the graph correctly" and
// "did we lower it correctly" become separately answerable questions. Set TSI_DUMP_GGML_IR=1 to see
// the intermediate.
#include "Builder.h"
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "Importer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

using namespace mlir;

namespace tsi::mlir_export {

std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    if (n_nodes == 0) {
        unsupported("graph has no nodes");
    }
    // A graph with no runtime args is legal when every leaf was baked in as a constant, so the
    // check is for no bound leafs at all, which would leave interior values undefined.
    if (opts.runtime_args.empty() && opts.const_leafs.empty()) {
        unsupported("graph has no runtime input tensors and no baked constants");
    }

    MLIRContext ctx;
    ctx.loadDialect<ggml::GgmlDialect, func::FuncDialect, arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, math::MathDialect>();

    // Empty `outputs` means the graph's single output, which is its last node. ggml graphs are
    // already topologically sorted, so the last node is the sink.
    std::vector<const ggml_tensor *> outs = opts.outputs;
    if (outs.empty()) {
        outs.push_back(ggml_graph_node(gf, n_nodes - 1));
    }

    OwningOpRef<ModuleOp> mod = importGraph(ctx, gf, opts, outs);

    if (failed(verify(*mod))) {
        // A dialect verifier rejected the imported graph, which means the graph violates one of
        // ggml's own invariants rather than merely being something we cannot lower.
        unsupported("the imported ggml-dialect module failed verification (diagnostics above)");
    }

    if (const char * dump = std::getenv("TSI_DUMP_GGML_IR")) {
        if (dump[0] == '1') {
            llvm::errs() << "--- ggml dialect (pre-lowering) ---\n";
            mod->print(llvm::errs());
            llvm::errs() << "-----------------------------------\n";
        }
    }

    convertGgmlToLinalg(*mod);

    if (failed(verify(*mod))) {
        unsupported("the lowered module failed MLIR verification (diagnostics above)");
    }

    std::string             out;
    llvm::raw_string_ostream os(out);
    mod->print(os);
    return out;
}

}  // namespace tsi::mlir_export
