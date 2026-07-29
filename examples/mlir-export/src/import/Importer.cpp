// ggml_cgraph -> ggml dialect, 1:1 with no lowering decisions.
#include "Importer.h"

#include "Builder.h"
#include "GgmlDialect.h"

#include <cstring>
#include <map>
#include <string>

using namespace mlir;

namespace tsi::mlir_export {

namespace {

// Reads a float out of op_params at the given int32 slot, the way ggml stores them.
float paramF32(const ggml_tensor * t, int slot) {
    float v;
    std::memcpy(&v, (const char *) t->op_params + slot * sizeof(int32_t), sizeof(float));
    return v;
}

int32_t paramI32(const ggml_tensor * t, int slot) {
    return ((const int32_t *) t->op_params)[slot];
}

class Importer {
  public:
    Importer(OpBuilder & b, Location loc, GraphBuilder & helpers) : b_(b), loc_(loc), h_(helpers) {}

    Value node(const ggml_tensor * n) {
        RankedTensorType ty = h_.tensorType(n);
        switch (n->op) {
            case GGML_OP_ADD:
                return ggml::AddOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]));
            case GGML_OP_MUL:
                return ggml::MulOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]));
            case GGML_OP_SCALE:
                return ggml::ScaleOp::create(b_, loc_, ty, val(n->src[0]), b_.getF32FloatAttr(paramF32(n, 0)),
                                             b_.getF32FloatAttr(paramF32(n, 1)));
            case GGML_OP_RMS_NORM:
                return ggml::RmsNormOp::create(b_, loc_, ty, val(n->src[0]), b_.getF32FloatAttr(paramF32(n, 0)));
            case GGML_OP_SOFT_MAX: {
                Value mask = n->src[1] ? val(n->src[1]) : Value();
                return ggml::SoftMaxOp::create(b_, loc_, ty, val(n->src[0]), mask,
                                               b_.getF32FloatAttr(paramF32(n, 0)),
                                               b_.getF32FloatAttr(paramF32(n, 1)));
            }
            case GGML_OP_MUL_MAT:
                return ggml::MulMatOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]));
            case GGML_OP_ROPE:
                return ggml::RopeOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]),
                                            b_.getI32IntegerAttr(paramI32(n, 1)),   // n_dims
                                            b_.getI32IntegerAttr(paramI32(n, 2)),   // mode
                                            b_.getF32FloatAttr(paramF32(n, 5)),     // freq_base
                                            b_.getF32FloatAttr(paramF32(n, 6)),     // freq_scale
                                            b_.getF32FloatAttr(paramF32(n, 7)),     // ext_factor
                                            b_.getF32FloatAttr(paramF32(n, 8)));    // attn_factor
            case GGML_OP_PERMUTE: {
                // op_params[0..3] verbatim, in ggml dim space.
                SmallVector<int32_t, 4> axes;
                for (int i = 0; i < 4; i++) {
                    axes.push_back(paramI32(n, i));
                }
                return ggml::PermuteOp::create(b_, loc_, ty, val(n->src[0]), b_.getDenseI32ArrayAttr(axes));
            }
            case GGML_OP_RESHAPE:
                return ggml::ReshapeOp::create(b_, loc_, ty, val(n->src[0]));
            case GGML_OP_CONT:
                return ggml::ContOp::create(b_, loc_, ty, val(n->src[0]));
            case GGML_OP_CONCAT:
                return ggml::ConcatOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]),
                                              b_.getI32IntegerAttr(paramI32(n, 0)));
            case GGML_OP_GET_ROWS:
                return ggml::GetRowsOp::create(b_, loc_, ty, val(n->src[0]), val(n->src[1]));
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(n) == GGML_UNARY_OP_SILU) {
                    return ggml::SiluOp::create(b_, loc_, ty, val(n->src[0]));
                }
                unsupported("no ggml dialect op for unary %s", ggml_unary_op_name(ggml_get_unary_op(n)));
            default:
                unsupported("no ggml dialect op for %s", ggml_op_name(n->op));
        }
    }

  private:
    Value val(const ggml_tensor * t) { return h_.valueOf(t); }

    OpBuilder &    b_;
    Location       loc_;
    GraphBuilder & h_;
};

}  // namespace

OwningOpRef<ModuleOp> importGraph(MLIRContext & ctx, ggml_cgraph * gf, const ExportOptions & opts,
                                  const std::vector<const ggml_tensor *> & outputs) {
    OpBuilder b(&ctx);
    Location  loc = b.getUnknownLoc();

    OwningOpRef<ModuleOp> mod = ModuleOp::create(loc);
    b.setInsertionPointToEnd(mod->getBody());

    GraphBuilder h(b, loc);

    SmallVector<Type> argTys;
    for (const ggml_tensor * t : opts.runtime_args) {
        argTys.push_back(h.tensorType(t));
    }
    SmallVector<Type> resTys;
    for (const ggml_tensor * t : outputs) {
        resTys.push_back(h.tensorType(t));
    }

    auto fn = func::FuncOp::create(b, loc, opts.func_name, b.getFunctionType(argTys, resTys));
    // emit-c-interface appends the result out-params after the inputs, so the ciface arg order is
    // [runtime_args..., outputs...]. The host shims rely on that.
    fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    for (size_t i = 0; i < opts.runtime_args.size(); i++) {
        fn.setArgAttr(i, "txe.name", b.getStringAttr("input_" + std::to_string(i)));
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        fn.setResultAttr(i, "txe.name", b.getStringAttr("res_" + std::to_string(i)));
    }

    Block * body = fn.addEntryBlock();
    b.setInsertionPointToEnd(body);

    for (size_t i = 0; i < opts.runtime_args.size(); i++) {
        h.setValue(opts.runtime_args[i], body->getArgument(i));
    }
    for (const ggml_tensor * leaf : opts.const_leafs) {
        h.setValue(leaf, h.bakedConstant(leaf));
    }

    Importer imp(b, loc, h);
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * n = ggml_graph_node(gf, i);
        h.setValue(n, imp.node(n));
    }

    SmallVector<Value> rets;
    for (const ggml_tensor * t : outputs) {
        rets.push_back(h.valueOf(t));
    }
    func::ReturnOp::create(b, loc, rets);

    return mod;
}

}  // namespace tsi::mlir_export
