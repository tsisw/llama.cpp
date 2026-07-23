#pragma once
// Rebuild a cache-free prefill graph from llama.cpp's live forward cgraph.
//
// The live graph runs attention through an in-place KV cache (SET_ROWS writes cache_k/v, later
// VIEWs alias the same buffer), which a pure-tensor MLIR function can't express. For a single
// prefill over an empty cache, attending to the cache is the same as attending to Kcur/Vcur, so we
// rebuild the same math cache-free with build_layer (only the op set the exporter lowers to FPGA).
//
// Weights come from g_wcap (captured while llama computed, see tsi_wholegraph.cpp); the prompt
// tokens, positions and rope/softmax params come from the live graph, so the reconstruction can be
// diffed against llama's own per-op logits.
#include "exporter.h"      // case_result, discover_leafs, build_func_text_baked, mlir_export_error
#include "model_layer.h"   // build_layer, LayerW, REAL_RMS_EPS

#include <cstring>
#include <map>
#include <string>
#include <vector>

// Weight data captured during llama's compute (defined in tsi_wholegraph.cpp). Keyed by core name,
// empty until the graph has run once.
extern std::map<std::string, std::vector<float>> g_wcap;

// Drop the scheduler's split annotations: "CPU#blk.0.attn_q.weight#0" -> "blk.0.attn_q.weight".
static std::string wg_core_name(const char * raw) {
    std::string s = raw ? raw : "";
    auto h1 = s.find('#');
    if (h1 != std::string::npos) s = s.substr(h1 + 1);   // leading "BACKEND#"
    auto h2 = s.rfind('#');
    if (h2 != std::string::npos) s = s.substr(0, h2);     // trailing "#<copy-index>"
    return s;
}

// Index every tensor in the live graph by core name, preferring the original leaf when a name has
// several tensors (the scheduler adds backend copies like "CPU#w#0").
static std::map<std::string, ggml_tensor *> wg_index_live(struct ggml_cgraph * live) {
    std::map<std::string, ggml_tensor *> m;
    auto is_original = [](const ggml_tensor * t) {
        return t->op == GGML_OP_NONE && std::string(t->name).find('#') == std::string::npos;
    };
    const int n = ggml_graph_n_nodes(live);
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        auto add = [&](ggml_tensor * t) {
            if (!t) return;
            std::string c = wg_core_name(t->name);
            if (c.empty()) return;
            auto it = m.find(c);
            if (it == m.end()) { m[c] = t; return; }
            if (is_original(t) && !is_original(it->second)) it->second = t;
        };
        add(nd);
        for (int s = 0; s < GGML_MAX_SRC; s++) add(nd->src[s]);
    }
    return m;
}

// Reconstruct the cache-free prefill graph. Throws mlir_export_error if the live graph isn't the
// expected llama shape, so capture can skip non-target graphs gracefully.
static case_result build_cachefree_from_live(struct ggml_cgraph * live) {
    std::map<std::string, ggml_tensor *> idx = wg_index_live(live);

    auto is_orig = [](const ggml_tensor * t) {
        return t && t->op == GGML_OP_NONE && t->data && std::string(t->name).find('#') == std::string::npos;
    };
    // Locate a weight/input tensor by core name. Prefer the persistent original leaf; otherwise take
    // the operand of a real compute op, then any match.
    auto need = [&](const std::string & core) -> ggml_tensor * {
        auto it = idx.find(core);
        if (it != idx.end() && is_orig(it->second)) return it->second;
        ggml_tensor * any = nullptr;
        const int nn = ggml_graph_n_nodes(live);
        for (int i = 0; i < nn; i++) {
            ggml_tensor * nd = ggml_graph_node(live, i);
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                ggml_tensor * sc = nd->src[s];
                if (!sc || wg_core_name(sc->name) != core) continue;
                if (nd->op == GGML_OP_MUL_MAT || nd->op == GGML_OP_MUL || nd->op == GGML_OP_GET_ROWS) return sc;
                if (!any) any = sc;
            }
        }
        if (any) return any;
        if (it != idx.end() && it->second) return it->second;
        throw mlir_export_error("live graph missing tensor '" + core + "'");
    };

    ggml_tensor * embd_t    = need("token_embd.weight");
    ggml_tensor * outnorm_t = need("output_norm.weight");
    ggml_tensor * outw_t    = need("output.weight");

    const int HIDDEN  = (int) embd_t->ne[0];
    const int N_VOCAB = (int) embd_t->ne[1];

    int N_LAYERS = 0;
    while (idx.count("blk." + std::to_string(N_LAYERS) + ".attn_q.weight")) N_LAYERS++;
    if (N_LAYERS == 0) throw mlir_export_error("live graph has no blk.*.attn_q.weight layers");

    // token ids: the embedding GET_ROWS has src1 = i32 ids[n_tokens]
    ggml_tensor * ids_live = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(live) && !ids_live; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_GET_ROWS && nd->src[0] &&
            wg_core_name(nd->src[0]->name) == "token_embd.weight" &&
            nd->src[1] && nd->src[1]->type == GGML_TYPE_I32) {
            ids_live = nd->src[1];
        }
    }
    if (!ids_live) throw mlir_export_error("live graph: could not find token-id input");
    { auto it = idx.find(wg_core_name(ids_live->name)); if (it != idx.end()) ids_live = it->second; }
    const int N_TOKENS = (int) ids_live->ne[0];

    // head_dim and the exact rope config from the first ROPE node (src0 = [head_dim, n_head, n_tok]).
    ggml_tensor * pos_live = nullptr;
    int HEAD_DIM = 0;
    for (int i = 0; i < ggml_graph_n_nodes(live) && !pos_live; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_ROPE) {
            HEAD_DIM  = (int) nd->src[0]->ne[0];
            pos_live  = nd->src[1];
            const int32_t * pi = (const int32_t *) nd->op_params;
            REAL_ROPE.n_dims     = pi[1];
            REAL_ROPE.mode       = pi[2];
            REAL_ROPE.n_ctx_orig = pi[4];
            memcpy(&REAL_ROPE.freq_base,   (const float *) nd->op_params + 5, sizeof(float));
            memcpy(&REAL_ROPE.freq_scale,  (const float *) nd->op_params + 6, sizeof(float));
            memcpy(&REAL_ROPE.ext_factor,  (const float *) nd->op_params + 7, sizeof(float));
            memcpy(&REAL_ROPE.attn_factor, (const float *) nd->op_params + 8, sizeof(float));
            memcpy(&REAL_ROPE.beta_fast,   (const float *) nd->op_params + 9, sizeof(float));
            memcpy(&REAL_ROPE.beta_slow,   (const float *) nd->op_params + 10, sizeof(float));
        }
    }
    if (!pos_live || HEAD_DIM == 0) throw mlir_export_error("live graph: could not find ROPE/positions");

    // attention scale from the first SOFT_MAX (op_params[0])
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_SOFT_MAX) { memcpy(&REAL_KQ_SCALE, nd->op_params, sizeof(float)); break; }
    }

    ggml_tensor * wq0 = need("blk.0.attn_q.weight");
    ggml_tensor * wk0 = need("blk.0.attn_k.weight");
    ggml_tensor * g0  = need("blk.0.ffn_gate.weight");
    const int N_HEAD    = (int) wq0->ne[1] / HEAD_DIM;
    const int N_HEAD_KV = (int) wk0->ne[1] / HEAD_DIM;
    const int INTER     = (int) g0->ne[1];

    // rms eps from the first RMS_NORM (op_params[0])
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_RMS_NORM) { memcpy(&REAL_RMS_EPS, nd->op_params, sizeof(float)); break; }
    }

    fprintf(stderr,
            "[tsi-wholegraph] live dims: layers=%d hidden=%d vocab=%d n_head=%d n_head_kv=%d "
            "head_dim=%d inter=%d n_tokens=%d eps=%g\n"
            "[tsi-wholegraph] rope: n_dims=%d mode=%d freq_base=%g freq_scale=%g ext=%g attn=%g "
            "beta_fast=%g beta_slow=%g n_ctx_orig=%d  kq_scale=%g\n",
            N_LAYERS, HIDDEN, N_VOCAB, N_HEAD, N_HEAD_KV, HEAD_DIM, INTER, N_TOKENS, REAL_RMS_EPS,
            REAL_ROPE.n_dims, REAL_ROPE.mode, REAL_ROPE.freq_base, REAL_ROPE.freq_scale,
            REAL_ROPE.ext_factor, REAL_ROPE.attn_factor, REAL_ROPE.beta_fast, REAL_ROPE.beta_slow,
            REAL_ROPE.n_ctx_orig, REAL_KQ_SCALE);

    size_t ctx_size = (size_t) 256 << 20;   // activations + ids/pos/mask + weight copies
    struct ggml_init_params params { /*.mem_size =*/ ctx_size, /*.mem_buffer =*/ NULL, /*.no_alloc =*/ false };

    case_result r;
    r.ctx = ggml_init(params);

    // Copy a weight into a fresh contiguous tensor. Use the captured data in g_wcap; fall back to the
    // live buffer when nothing was captured yet (at capture time only the shape is needed). The live
    // scheduler copies are recycled scratch after compute and q/k/v have no persistent leaf, so
    // referencing them directly would feed stale data into the graph.
    auto bind_w = [&](const std::string & core, ggml_tensor * shape_src) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor(r.ctx, GGML_TYPE_F32, ggml_n_dims(shape_src), shape_src->ne);
        auto it = g_wcap.find(core);
        if (it != g_wcap.end() && it->second.size() == (size_t) ggml_nelements(t)) {
            memcpy(t->data, it->second.data(), ggml_nbytes(t));
        } else if (shape_src->data) {
            memcpy(t->data, shape_src->data, ggml_nbytes(t));
        }
        ggml_set_name(t, core.c_str());
        return t;
    };
    embd_t    = bind_w("token_embd.weight",  embd_t);
    outnorm_t = bind_w("output_norm.weight", outnorm_t);
    outw_t    = bind_w("output.weight",      outw_t);

    struct ggml_tensor * ids = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, N_TOKENS);
    memcpy(ids->data, ids_live->data, ggml_nbytes(ids));
    // fresh prefill positions 0..n-1 (the live pos buffer is recycled after compute)
    struct ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, N_TOKENS);
    for (int i = 0; i < N_TOKENS; i++) ((int32_t *) pos->data)[i] = i;

    fprintf(stderr, "[tsi-wholegraph] ids:");
    for (int i = 0; i < N_TOKENS; i++) fprintf(stderr, " %d", ((const int32_t *) ids->data)[i]);
    fprintf(stderr, "   pos:");
    for (int i = 0; i < N_TOKENS; i++) fprintf(stderr, " %d", ((const int32_t *) pos->data)[i]);
    fprintf(stderr, "\n");

    struct ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, N_TOKENS, N_TOKENS);
    {
        float * md = (float *) mask->data;
        for (int q = 0; q < N_TOKENS; q++)
            for (int k = 0; k < N_TOKENS; k++)
                md[q * N_TOKENS + k] = (k <= q) ? 0.0f : -INFINITY;   // causal
    }

    struct ggml_tensor * cur = ggml_get_rows(r.ctx, embd_t, ids);
    for (int il = 0; il < N_LAYERS; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        LayerW lw;
        lw.attn_norm = bind_w(p + "attn_norm.weight",   need(p + "attn_norm.weight"));
        lw.wq        = bind_w(p + "attn_q.weight",       need(p + "attn_q.weight"));
        lw.wk        = bind_w(p + "attn_k.weight",       need(p + "attn_k.weight"));
        lw.wv        = bind_w(p + "attn_v.weight",       need(p + "attn_v.weight"));
        lw.wo        = bind_w(p + "attn_output.weight",  need(p + "attn_output.weight"));
        lw.ffn_norm  = bind_w(p + "ffn_norm.weight",     need(p + "ffn_norm.weight"));
        lw.gate      = bind_w(p + "ffn_gate.weight",     need(p + "ffn_gate.weight"));
        lw.up        = bind_w(p + "ffn_up.weight",       need(p + "ffn_up.weight"));
        lw.down      = bind_w(p + "ffn_down.weight",     need(p + "ffn_down.weight"));
        cur = build_layer(r.ctx, cur, lw, pos, mask, HEAD_DIM, N_HEAD, N_HEAD_KV, N_TOKENS);
    }
    struct ggml_tensor * normed_final = ggml_mul(r.ctx, ggml_rms_norm(r.ctx, cur, REAL_RMS_EPS), outnorm_t);
    struct ggml_tensor * logits       = ggml_mul_mat(r.ctx, outw_t, normed_final);   // [N_VOCAB, N_TOKENS]

    r.gf = ggml_new_graph(r.ctx);
    ggml_build_forward_expand(r.gf, logits);

    r.leafs = discover_leafs(r.gf);
    r.runtime_args = r.leafs;   // every leaf is a runtime arg; nothing baked
    r.func_text = build_func_text_baked(r.gf, "forward", r.runtime_args, {});
    return r;
}
