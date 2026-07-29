// Leaf discovery and the ggml-op -> emitter dispatch.
#include "Builder.h"

#include <map>

using namespace mlir;

namespace tsi::mlir_export {

std::vector<const ggml_tensor *> discoverLeafs(ggml_cgraph * gf) {
    // Walks src[] on the public ggml_tensor rather than the graph's leafs[]/n_leafs, which are
    // private. First-seen order, so it is stable for a given graph.
    std::vector<const ggml_tensor *>   leafs;
    std::map<const ggml_tensor *, int> seen;
    const int                          n_nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                continue;
            }
            if (src->op == GGML_OP_NONE && seen.find(src) == seen.end()) {
                seen[src] = (int) leafs.size();
                leafs.push_back(src);
            }
        }
    }
    return leafs;
}

Value GraphBuilder::emitNode(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return emitMulMat(node);
        case GGML_OP_ADD:
            return emitAdd(node);
        case GGML_OP_MUL:
            return emitMul(node);
        case GGML_OP_SCALE:
            return emitScale(node);
        case GGML_OP_RMS_NORM:
            return emitRmsNorm(node);
        case GGML_OP_SOFT_MAX:
            return emitSoftMax(node);
        case GGML_OP_ROPE:
            return emitRope(node);
        case GGML_OP_PERMUTE:
            return emitPermute(node);
        case GGML_OP_RESHAPE:
        case GGML_OP_CONT:
            return emitReshapeLike(node, node->src[0]);
        case GGML_OP_GET_ROWS:
            return emitGetRows(node);
        case GGML_OP_CONCAT:
            return emitConcat(node);
        case GGML_OP_UNARY:
            if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                return emitSilu(node);
            }
            unsupported("unsupported unary op: %s", ggml_unary_op_name(ggml_get_unary_op(node)));
        default:
            unsupported("unsupported op: %s", ggml_op_name(node->op));
    }
}

}  // namespace tsi::mlir_export
