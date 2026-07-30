#pragma once
// Read llama's live KV cache, so a reconstructed decode graph can start from the state llama is
// actually in.
//
// llama holds one buffer per layer per kind, named `cache_k_l<il>` / `cache_v_l<il>`. Measured on
// SmolLM2-135M (30 layers, head_dim 64, 3 KV heads, n_ctx 4096):
//
//   cache_k_l0 : f16 [192, 4096]        <- the persistent leaf; ne[0] = head_dim * n_head_kv
//   VIEW       : f16 [64, 3, 256]       <- what a decode step reads; ne[2] = n_kv, the live window
//   PERMUTE    : f16 [64, 256, 3]       <- fed to FLASH_ATTN_EXT
//
// Two facts make this cheap, and both were checked against a real graph rather than assumed:
//
//  1. **The layout already matches.** ne[0] packs one cell as head_dim-fastest then head, and cells
//     are ne[1]. So cell c sits at [c * n_embd_k_gqa, ...) and the first L cells reinterpreted as
//     [head_dim, n_head_kv, L] are exactly what build_decode wants. No shuffling.
//  2. **V is not transposed.** llama sets `attn_v_trans = !cparams.flash_attn`, and flash attention is
//     on by default, so cache_v has the same shape as cache_k. A build with `-fa off` would transpose
//     V and needs handling; extractLayer refuses that case rather than reading it wrong.
#include "tsi/graph/LiveGraphBuilder.h"   // wg_core_name, wg_index_live
#include "ggml.h"

#include <cstring>
#include <string>
#include <vector>

// What one kind of cache looks like in the live graph.
struct LiveCacheInfo {
    int n_layers  = 0;
    int head_dim  = 0;
    int n_head_kv = 0;
    int n_cells   = 0;   // n_ctx: the buffer's capacity
    int n_kv      = 0;   // the window a decode step actually reads, from the VIEW node
    bool found    = false;
};

// Locate `cache_k_l*` / `cache_v_l*` and read the geometry off them. `kind` is "k" or "v".
static inline LiveCacheInfo live_cache_probe(struct ggml_cgraph * live, const char * kind,
                                             int head_dim, int n_head_kv) {
    LiveCacheInfo info;
    info.head_dim  = head_dim;
    info.n_head_kv = n_head_kv;

    const std::string prefix = std::string("cache_") + kind + "_l";

    // The persistent leaf, and the VIEW that reads it. Located structurally: the leaf is the only
    // GGML_OP_NONE tensor with that name, and any VIEW over it carries the live window in ne[2].
    const int nn = ggml_graph_n_nodes(live);
    for (int i = 0; i < nn; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor * sc = nd->src[s];
            if (!sc || sc->op != GGML_OP_NONE || !sc->data) {
                continue;
            }
            const std::string cn = wg_core_name(sc->name);
            if (cn.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const int il = atoi(cn.c_str() + prefix.size());
            if (il + 1 > info.n_layers) {
                info.n_layers = il + 1;
            }
            info.n_cells = (int) sc->ne[1];
            info.found   = true;
        }
        if (nd->op == GGML_OP_VIEW && nd->src[0] &&
            wg_core_name(nd->src[0]->name).compare(0, prefix.size(), prefix) == 0 &&
            ggml_n_dims(nd) == 3) {
            info.n_kv = (int) nd->ne[2];
        }
    }
    return info;
}

// The layer's cache buffer: the persistent leaf, not one of its views.
static inline ggml_tensor * live_cache_leaf(struct ggml_cgraph * live, const char * kind, int il) {
    char nm[64];
    snprintf(nm, sizeof(nm), "cache_%s_l%d", kind, il);
    const int nn = ggml_graph_n_nodes(live);
    for (int i = 0; i < nn; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor * sc = nd->src[s];
            if (sc && sc->op == GGML_OP_NONE && sc->data && wg_core_name(sc->name) == nm) {
                return sc;
            }
        }
    }
    return nullptr;
}

// Materialize the first `cells` cells of layer il as a fresh f32 [head_dim, n_head_kv, cells] tensor
// in `ctx`, which is the shape build_decode consumes.
//
// f16 -> f32 here rather than keeping f16: the reconstruction computes in f32, and the exporter would
// widen it anyway (promoteGgmlToF32). Converting once on the host keeps the graph uniform. The
// narrowing back to f16 belongs with the cache write, not the read.
//
// Returns nullptr if the layer is missing or the layout is not the one documented above, rather than
// reading something plausible-looking out of the wrong offsets.
static inline ggml_tensor * live_cache_extract(struct ggml_context * ctx, struct ggml_cgraph * live,
                                               const char * kind, int il, const LiveCacheInfo & info,
                                               int cells) {
    ggml_tensor * src = live_cache_leaf(live, kind, il);
    if (!src) {
        return nullptr;
    }
    const int64_t per_cell = (int64_t) info.head_dim * info.n_head_kv;
    if (src->ne[0] != per_cell) {
        // Either a transposed V cache (-fa off) or a geometry we have not seen. Both would read
        // correctly-shaped garbage if we went ahead.
        fprintf(stderr, "[tsi-mlir] cache_%s_l%d: ne[0]=%lld, expected head_dim*n_head_kv=%lld. "
                        "A transposed V cache (built with -fa off) is not handled.\n",
                kind, il, (long long) src->ne[0], (long long) per_cell);
        return nullptr;
    }
    if (cells > (int) src->ne[1]) {
        return nullptr;
    }

    ggml_tensor * out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.head_dim, info.n_head_kv, cells);
    float *       d   = (float *) out->data;

    if (src->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) src->data;
        for (int64_t i = 0; i < per_cell * cells; i++) {
            d[i] = ggml_fp16_to_fp32(s[i]);
        }
    } else if (src->type == GGML_TYPE_F32) {
        memcpy(d, src->data, (size_t) per_cell * cells * sizeof(float));
    } else {
        fprintf(stderr, "[tsi-mlir] cache_%s_l%d: unsupported cache type %s\n", kind, il,
                ggml_type_name(src->type));
        return nullptr;
    }
    return out;
}
