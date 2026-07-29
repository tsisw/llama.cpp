#pragma once
// Shared model load + decode-graph builder for the KV-cache decode path. Used by decode_cpu_check,
// which both validates the graph on CPU and emits it as MLIR, so the checked and exported graphs are
// the same one by construction.
#include "tsi/graph/DecodeLayer.h"   // build_decode_layer, and via it model_layer.h (build_layer, wg_rope, REAL_*)
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// F32/quant row -> f32 (rows x cols, reading ne[0]-wide source rows).
static inline void dm_deq(const ggml_tensor * t, int rows, int cols, float * out) {
    if (t->type == GGML_TYPE_F32) {
        const float * src = (const float *) t->data;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) out[r * cols + c] = src[(size_t) r * t->ne[0] + c];
        return;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    size_t rb = ggml_row_size(t->type, t->ne[0]);
    std::vector<float> row(t->ne[0]);
    for (int r = 0; r < rows; r++) {
        tt->to_float((const uint8_t *) t->data + (size_t) r * rb, row.data(), t->ne[0]);
        for (int c = 0; c < cols; c++) out[r * cols + c] = row[c];
    }
}
static inline ggml_tensor * dm_getT(ggml_context * c, const char * fmt, int il) {
    char nm[128]; snprintf(nm, sizeof(nm), fmt, il);
    ggml_tensor * t = ggml_get_tensor(c, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    return t;
}

struct DecodeModel {
    int n_layers = 0, hidden = 0, inter = 0, n_head = 0, n_head_kv = 0, head_dim = 0, hidden_kv = 0, n_vocab = 0;
    ggml_context *      wc  = nullptr;                 // persistent context holding every weight tensor
    ggml_tensor *       embd = nullptr, * onw = nullptr, * oww = nullptr;
    std::vector<LayerW> lw;                            // per-layer weights (materialized f32 tensors)
};

// Load a GGUF, dequantize all weights into a persistent context, and set the REAL_* rope/rms globals.
static inline DecodeModel load_decode_model(const char * path) {
    ggml_context * gc = nullptr;
    gguf_init_params gp { false, &gc };
    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) { fprintf(stderr, "load fail %s\n", path); exit(1); }
    auto ru = [&](const char * k, int d) { int64_t i = gguf_find_key(gguf, k); return i < 0 ? d : (int) gguf_get_val_u32(gguf, i); };
    auto rf = [&](const char * k, float d) { int64_t i = gguf_find_key(gguf, k); return i < 0 ? d : gguf_get_val_f32(gguf, i); };

    DecodeModel M;
    M.n_layers  = ru("llama.block_count", 22);
    M.hidden    = ru("llama.embedding_length", 2048);
    M.inter     = ru("llama.feed_forward_length", 5632);
    M.n_head    = ru("llama.attention.head_count", 32);
    M.n_head_kv = ru("llama.attention.head_count_kv", 4);
    int key_len = ru("llama.attention.key_length", 0);
    M.head_dim  = key_len > 0 ? key_len : M.hidden / M.n_head;
    M.hidden_kv = M.head_dim * M.n_head_kv;
    REAL_RMS_EPS = rf("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    REAL_ROPE = RopeParams{}; REAL_ROPE.n_dims = M.head_dim; REAL_ROPE.mode = 0;
    REAL_ROPE.n_ctx_orig = ru("llama.context_length", 2048);
    REAL_ROPE.freq_base  = rf("llama.rope.freq_base", 10000.0f); REAL_ROPE.freq_scale = 1.0f;
    REAL_KQ_SCALE = 1.0f / sqrtf((float) M.head_dim);

    ggml_tensor * embd_t = ggml_get_tensor(gc, "token_embd.weight");
    ggml_tensor * on_t   = ggml_get_tensor(gc, "output_norm.weight");
    ggml_tensor * ow_t   = ggml_get_tensor(gc, "output.weight"); if (!ow_t) ow_t = embd_t;
    M.n_vocab = (int) embd_t->ne[1];
    const int HID = M.hidden, HKV = M.hidden_kv, INT = M.inter, VOC = M.n_vocab;

    std::vector<float> embd_d((size_t) VOC * HID), ow_d((size_t) VOC * HID), on_d(HID);
    dm_deq(embd_t, VOC, HID, embd_d.data()); dm_deq(ow_t, VOC, HID, ow_d.data());
    memcpy(on_d.data(), on_t->data, HID * sizeof(float));
    struct LW { std::vector<float> an, wq, wk, wv, wo, fn, g, u, d; };
    std::vector<LW> W(M.n_layers);
    for (int il = 0; il < M.n_layers; il++) {
        W[il].an.resize(HID); memcpy(W[il].an.data(), dm_getT(gc, "blk.%d.attn_norm.weight", il)->data, HID * sizeof(float));
        W[il].fn.resize(HID); memcpy(W[il].fn.data(), dm_getT(gc, "blk.%d.ffn_norm.weight", il)->data, HID * sizeof(float));
        W[il].wq.resize((size_t) HID * HID); dm_deq(dm_getT(gc, "blk.%d.attn_q.weight", il), HID, HID, W[il].wq.data());
        W[il].wk.resize((size_t) HKV * HID); dm_deq(dm_getT(gc, "blk.%d.attn_k.weight", il), HKV, HID, W[il].wk.data());
        W[il].wv.resize((size_t) HKV * HID); dm_deq(dm_getT(gc, "blk.%d.attn_v.weight", il), HKV, HID, W[il].wv.data());
        W[il].wo.resize((size_t) HID * HID); dm_deq(dm_getT(gc, "blk.%d.attn_output.weight", il), HID, HID, W[il].wo.data());
        W[il].g.resize((size_t) INT * HID);  dm_deq(dm_getT(gc, "blk.%d.ffn_gate.weight", il), INT, HID, W[il].g.data());
        W[il].u.resize((size_t) INT * HID);  dm_deq(dm_getT(gc, "blk.%d.ffn_up.weight", il),   INT, HID, W[il].u.data());
        W[il].d.resize((size_t) HID * INT);  dm_deq(dm_getT(gc, "blk.%d.ffn_down.weight", il), HID, INT, W[il].d.data());
    }
    ggml_free(gc); gguf_free(gguf);

    size_t wb = (size_t) 2 * VOC * HID * 4 +
                (size_t) M.n_layers * (2 * HID * HID + 2 * HKV * HID + 3 * INT * HID) * 4;
    ggml_init_params wp { wb + wb / 4 + ((size_t) 64 << 20), nullptr, false };
    M.wc = ggml_init(wp);
    M.embd = ggml_new_tensor_2d(M.wc, GGML_TYPE_F32, HID, VOC); memcpy(M.embd->data, embd_d.data(), ggml_nbytes(M.embd));
    M.onw  = ggml_new_tensor_1d(M.wc, GGML_TYPE_F32, HID);      memcpy(M.onw->data,  on_d.data(),  ggml_nbytes(M.onw));
    M.oww  = ggml_new_tensor_2d(M.wc, GGML_TYPE_F32, HID, VOC); memcpy(M.oww->data,  ow_d.data(),  ggml_nbytes(M.oww));
    M.lw.resize(M.n_layers);
    auto mk1 = [&](std::vector<float> & s) { ggml_tensor * t = ggml_new_tensor_1d(M.wc, GGML_TYPE_F32, (int64_t) s.size()); memcpy(t->data, s.data(), ggml_nbytes(t)); return t; };
    auto mk2 = [&](std::vector<float> & s, int a, int b) { ggml_tensor * t = ggml_new_tensor_2d(M.wc, GGML_TYPE_F32, a, b); memcpy(t->data, s.data(), ggml_nbytes(t)); return t; };
    for (int il = 0; il < M.n_layers; il++) {
        M.lw[il].attn_norm = mk1(W[il].an); M.lw[il].ffn_norm = mk1(W[il].fn);
        M.lw[il].wq = mk2(W[il].wq, HID, HID);  M.lw[il].wk = mk2(W[il].wk, HID, HKV);
        M.lw[il].wv = mk2(W[il].wv, HID, HKV);  M.lw[il].wo = mk2(W[il].wo, HID, HID);
        M.lw[il].gate = mk2(W[il].g, HID, INT); M.lw[il].up = mk2(W[il].u, HID, INT); M.lw[il].down = mk2(W[il].d, INT, HID);
    }
    return M;
}

// logits = output * (rms_norm(x) * output_norm); the per-layer decode chain is build_decode_layer.
static inline ggml_tensor * build_decode(ggml_context * ctx, const DecodeModel & M,
        ggml_tensor * id, ggml_tensor * pos, ggml_tensor * mask,
        const std::vector<ggml_tensor *> & cacheK, const std::vector<ggml_tensor *> & cacheV,
        std::vector<ggml_tensor *> & knew, std::vector<ggml_tensor *> & vnew) {
    knew.assign(M.n_layers, nullptr); vnew.assign(M.n_layers, nullptr);
    ggml_tensor * x = ggml_get_rows(ctx, M.embd, id);
    for (int il = 0; il < M.n_layers; il++)
        x = build_decode_layer(ctx, x, M.lw[il], pos, cacheK[il], cacheV[il], mask,
                               M.head_dim, M.n_head, M.n_head_kv, &knew[il], &vnew[il]);
    return ggml_mul_mat(ctx, M.oww, ggml_mul(ctx, ggml_rms_norm(ctx, x, REAL_RMS_EPS), M.onw));
}
