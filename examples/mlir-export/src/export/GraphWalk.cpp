// Leaf discovery. The ggml-op dispatch now lives in the importer (src/import/Importer.cpp).
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

}  // namespace tsi::mlir_export
