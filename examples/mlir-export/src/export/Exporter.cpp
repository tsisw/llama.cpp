// Module and function assembly: sets up the MLIR context, builds func @<name> with the TSI
// entry-point attributes, seeds arguments and baked constants, dispatches every graph node,
// verifies, and prints.
#include "Builder.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace tsi::mlir_export {

std::string exportGraph(ggml_cgraph * gf, const ExportOptions & opts) {
    const int n_nodes = ggml_graph_n_nodes(gf);
    if (n_nodes == 0) {
        unsupported("graph has no nodes");
    }
    if (opts.runtime_args.empty()) {
        unsupported("graph has no runtime input tensors");
    }

    MLIRContext ctx;
    ctx.loadDialect<func::FuncDialect, arith::ArithDialect, tensor::TensorDialect, linalg::LinalgDialect,
                    math::MathDialect>();

    OpBuilder b(&ctx);
    Location  loc = b.getUnknownLoc();

    OwningOpRef<ModuleOp> mod = ModuleOp::create(loc);
    b.setInsertionPointToEnd(mod->getBody());

    GraphBuilder gb(b, loc);

    // Empty `outputs` means the graph's single output, which is its last node. ggml graphs are
    // already topologically sorted, so the last node is the sink.
    std::vector<const ggml_tensor *> outs = opts.outputs;
    if (outs.empty()) {
        outs.push_back(ggml_graph_node(gf, n_nodes - 1));
    }

    SmallVector<Type> argTys;
    for (const ggml_tensor * t : opts.runtime_args) {
        argTys.push_back(gb.tensorType(t));
    }
    SmallVector<Type> resTys;
    for (const ggml_tensor * t : outs) {
        resTys.push_back(gb.tensorType(t));
    }

    auto fn = func::FuncOp::create(b, loc, opts.func_name, b.getFunctionType(argTys, resTys));
    // emit-c-interface appends the result out-params after the inputs, so the ciface arg order is
    // [runtime_args..., outputs...]. The host shims rely on that.
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    for (size_t i = 0; i < opts.runtime_args.size(); i++) {
        fn.setArgAttr(i, "txe.name", b.getStringAttr("input_" + std::to_string(i)));
    }
    for (size_t i = 0; i < outs.size(); i++) {
        fn.setResultAttr(i, "txe.name", b.getStringAttr("res_" + std::to_string(i)));
    }

    Block * body = fn.addEntryBlock();
    b.setInsertionPointToEnd(body);

    for (size_t i = 0; i < opts.runtime_args.size(); i++) {
        gb.setValue(opts.runtime_args[i], body->getArgument(i));
    }
    for (const ggml_tensor * leaf : opts.const_leafs) {
        gb.setValue(leaf, gb.bakedConstant(leaf));
    }

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        gb.setValue(node, gb.emitNode(node));
    }

    SmallVector<Value> rets;
    for (const ggml_tensor * t : outs) {
        rets.push_back(gb.valueOf(t));
    }
    func::ReturnOp::create(b, loc, rets);

    // The whole point of building through the API: a structurally invalid graph fails here, with
    // MLIR's own diagnostic naming the op, instead of as an opaque parse error inside the Python
    // compiler driver several stages later.
    if (failed(verify(*mod))) {
        unsupported("the built module failed MLIR verification (diagnostics above)");
    }

    std::string out;
    llvm::raw_string_ostream os(out);
    mod->print(os);
    return out;
}

}  // namespace tsi::mlir_export
