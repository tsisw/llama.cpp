// Leaf discovery and weight/input classification. The ggml-op dispatch now lives in the importer
// (src/import/Importer.cpp).
#include "Builder.h"

#include <map>
#include <string>

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

// Strip the scheduler's decorations from a tensor name: a leading "BACKEND#" and a trailing
// "#<copy-index>". Mirrors wg_core_name in LiveGraphBuilder.h; duplicated rather than shared
// because that header is llama-specific and this library must not depend on it.
static std::string core_name(const char * raw) {
    std::string s = raw ? raw : "";
    auto h1 = s.find('#');
    if (h1 != std::string::npos) {
        s = s.substr(h1 + 1);
    }
    auto h2 = s.rfind('#');
    if (h2 != std::string::npos) {
        s = s.substr(0, h2);
    }
    return s;
}

bool isModelWeight(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr) {
        return false;   // no data to bake
    }
    // Only f32/i32 can be baked; bakedConstant reads those two element types.
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_I32) {
        return false;
    }
    const std::string cn = core_name(t->name);
    return cn.size() >= 7 && cn.compare(cn.size() - 7, 7, ".weight") == 0;
}

void partitionWeights(const std::vector<const ggml_tensor *> & leafs,
                      std::vector<const ggml_tensor *> & args,
                      std::vector<const ggml_tensor *> & consts) {
    for (const ggml_tensor * t : leafs) {
        if (isModelWeight(t)) {
            consts.push_back(t);
        } else {
            args.push_back(t);
        }
    }
}

}  // namespace tsi::mlir_export
