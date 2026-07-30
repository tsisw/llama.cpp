// The public entry point: import the ggml graph into the ggml dialect, lower it to linalg, verify,
// print.
//
// Splitting import from lowering is the point of the dialect: "did we read the graph correctly" and
// "did we lower it correctly" become separately answerable questions. Set TSI_DUMP_GGML_IR=1 to see
// the intermediate.
#include "IRBuilder.h"
#include "GgmlDialect.h"
#include "GgmlToLinalg.h"
#include "Importer.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace mlir;

namespace tsi::mlir_export {

// Every discovered leaf that is not a declared runtime arg and not a cache read stands for a
// constant. Weights therefore need no flag and no name heuristic: declaring the per-step inputs is
// what makes everything else a constant.
static std::vector<const ggml_tensor *> constLeafs(ggml_cgraph * gf, const ExportOptions & opts) {
    std::set<const ggml_tensor *> bound(opts.runtime_args.begin(), opts.runtime_args.end());
    for (const CacheSpec & c : opts.caches) {
        bound.insert(c.read.begin(), c.read.end());
    }

    std::vector<const ggml_tensor *> consts;
    for (const ggml_tensor * leaf : discoverLeafs(gf)) {
        if (bound.count(leaf) == 0) {
            consts.push_back(leaf);
        }
    }
    return consts;
}

std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    if (n_nodes == 0) {
        unsupported("graph has no nodes");
    }

    const std::vector<const ggml_tensor *> consts = constLeafs(gf, opts);
    // A graph with no runtime args is legal when every leaf was baked in as a constant, so the
    // check is for no bound leafs at all, which would leave interior values undefined.
    if (opts.runtime_args.empty() && consts.empty() && opts.caches.empty()) {
        unsupported("graph has no runtime input tensors, constants or caches");
    }

    MLIRContext ctx;
    // memref/bufferization are for the KV cache only; see CacheSpec in Exporter.h.
    ctx.loadDialect<ggml::GgmlDialect, func::FuncDialect, arith::ArithDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect>();

    // Empty `outputs` means the graph's single output, which is its last node. ggml graphs are
    // already topologically sorted, so the last node is the sink.
    std::vector<const ggml_tensor *> outs = opts.outputs;
    if (outs.empty()) {
        outs.push_back(ggml_graph_node(gf, n_nodes - 1));
    }

    OwningOpRef<ModuleOp> mod = importGraph(ctx, gf, opts, outs, consts);

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

    // f16/bf16 -> f32 arithmetic, before lowering, so the patterns only ever see f32 and every
    // reduction accumulates in f32. No-op on an all-f32 graph.
    promoteGgmlToF32(*mod);

    if (const char * dump = std::getenv("TSI_DUMP_GGML_IR")) {
        if (dump[0] == '1') {
            llvm::errs() << "--- ggml dialect (promoted to f32) ---\n";
            mod->print(llvm::errs());
            llvm::errs() << "--------------------------------------\n";
        }
    }

    convertGgmlToLinalg(*mod);

    if (failed(verify(*mod))) {
        unsupported("the lowered module failed MLIR verification (diagnostics above)");
    }

    std::string              out;
    llvm::raw_string_ostream os(out);
    if (opts.format == Format::Bytecode) {
        // Binary. Resource blobs stay raw bytes here; printing would hex them at 2x the size.
        if (failed(writeBytecodeToFile(*mod, os))) {
            unsupported("writing MLIR bytecode failed");
        }
    } else {
        mod->print(os);
    }
    return out;
}

}  // namespace tsi::mlir_export
