// CPU check for build_layer, no FPGA needed. Rebuilds the cache-free forward from a GGUF and runs
// it with ggml_graph_compute_with_ctx (no gallocr reuse, so intermediates stay readable), then
// prints the next-token argmax per column and each layer's output. Compare the argmax to ref_check.
//
// Usage: recon_cpu_check <model.gguf> <id0> [id1 ...]   (ids from ref_check)
#include "model_layer.h"      // build_layer, LayerW, REAL_RMS_EPS, REAL_ROPE, REAL_KQ_SCALE
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void deq(const ggml_tensor * t, int rows, int cols, float * out) {
    if (t->type == GGML_TYPE_F32) {                     // F32 model: to_float is NULL, copy directly
        const float * src = (const float *) t->data;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) out[r * cols + c] = src[(size_t) r * t->ne[0] + c];
        return;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
    std::vector<float> row(t->ne[0]);
    for (int r = 0; r < rows; r++) {
        tt->to_float((const uint8_t *) t->data + (size_t) r * row_bytes, row.data(), t->ne[0]);
        for (int c = 0; c < cols; c++) out[r * cols + c] = row[c];
    }
}

static ggml_tensor * getT(ggml_context * c, const char * fmt, int il) {
    char nm[128]; snprintf(nm, sizeof(nm), fmt, il);
    ggml_tensor * t = ggml_get_tensor(c, nm);
    if (!t) { fprintf(stderr, "missing %s\n", nm); exit(1); }
    return t;
}

static void argmax_cols(const float * logits, int vocab, int ntok) {
    fprintf(stderr, "argmax per column:");
    for (int t = 0; t < ntok; t++) {
        int best = 0; float bv = logits[(size_t) t * vocab];
        for (int v = 1; v < vocab; v++) { float x = logits[(size_t) t * vocab + v]; if (x > bv) { bv = x; best = v; } }
        fprintf(stderr, " %d(%.4f)", best, bv);
    }
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf id0 [id1 ...]\n", argv[0]); return 1; }
    const char * path = argv[1];
    std::vector<int32_t> ids;
    for (int i = 2; i < argc; i++) ids.push_back(atoi(argv[i]));
    const int N_TOKENS = (int) ids.size();

    ggml_context * gc = nullptr;
    gguf_init_params gp { /*.no_alloc=*/false, /*.ctx=*/&gc };
    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) { fprintf(stderr, "load fail %s\n", path); return 1; }

    auto ru = [&](const char * k, int d) { int64_t i = gguf_find_key(gguf, k); return i < 0 ? d : (int) gguf_get_val_u32(gguf, i); };
    auto rf = [&](const char * k, float d) { int64_t i = gguf_find_key(gguf, k); return i < 0 ? d : gguf_get_val_f32(gguf, i); };
    const int   N_LAYERS  = ru("llama.block_count", 22);
    const int   HIDDEN    = ru("llama.embedding_length", 2048);
    const int   INTER     = ru("llama.feed_forward_length", 5632);
    const int   N_HEAD    = ru("llama.attention.head_count", 32);
    const int   N_HEAD_KV = ru("llama.attention.head_count_kv", 4);
    const int   KEY_LEN   = ru("llama.attention.key_length", 0);
    const int   HEAD_DIM  = KEY_LEN > 0 ? KEY_LEN : HIDDEN / N_HEAD;
    const int   HIDDEN_KV = HEAD_DIM * N_HEAD_KV;
    const float FREQ_BASE = rf("llama.rope.freq_base", 10000.0f);
    REAL_RMS_EPS = rf("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    REAL_ROPE = RopeParams{};
    REAL_ROPE.n_dims     = HEAD_DIM;
    REAL_ROPE.mode       = 0;               // NORMAL (flip to test NEOX=2)
    REAL_ROPE.n_ctx_orig = ru("llama.context_length", 2048);
    REAL_ROPE.freq_base  = FREQ_BASE;
    REAL_ROPE.freq_scale = 1.0f;
    REAL_KQ_SCALE = 1.0f / sqrtf((float) HEAD_DIM);
    if (getenv("ROPE_NEOX")) REAL_ROPE.mode = 2;

    fprintf(stderr, "dims: layers=%d hidden=%d inter=%d n_head=%d n_head_kv=%d head_dim=%d eps=%g freq_base=%g mode=%d ntok=%d\n",
            N_LAYERS, HIDDEN, INTER, N_HEAD, N_HEAD_KV, HEAD_DIM, REAL_RMS_EPS, FREQ_BASE, REAL_ROPE.mode, N_TOKENS);

    ggml_tensor * embd_t    = ggml_get_tensor(gc, "token_embd.weight");
    ggml_tensor * outnorm_t = ggml_get_tensor(gc, "output_norm.weight");
    ggml_tensor * outw_t    = ggml_get_tensor(gc, "output.weight");
    if (!outw_t) outw_t = embd_t;   // tied embeddings
    const int N_VOCAB = (int) embd_t->ne[1];

    // dequantize everything to f32 host buffers
    std::vector<float> embd_d((size_t) N_VOCAB * HIDDEN), outw_d((size_t) N_VOCAB * HIDDEN), outnorm_d(HIDDEN);
    deq(embd_t, N_VOCAB, HIDDEN, embd_d.data());
    deq(outw_t, N_VOCAB, HIDDEN, outw_d.data());
    memcpy(outnorm_d.data(), outnorm_t->data, HIDDEN * sizeof(float));

    struct LW { std::vector<float> an, wq, wk, wv, wo, fn, g, u, d; };
    std::vector<LW> W(N_LAYERS);
    for (int il = 0; il < N_LAYERS; il++) {
        W[il].an.resize(HIDDEN);                     memcpy(W[il].an.data(), getT(gc, "blk.%d.attn_norm.weight", il)->data, HIDDEN * sizeof(float));
        W[il].fn.resize(HIDDEN);                     memcpy(W[il].fn.data(), getT(gc, "blk.%d.ffn_norm.weight", il)->data, HIDDEN * sizeof(float));
        W[il].wq.resize((size_t) HIDDEN * HIDDEN);    deq(getT(gc, "blk.%d.attn_q.weight", il), HIDDEN,    HIDDEN, W[il].wq.data());
        W[il].wk.resize((size_t) HIDDEN_KV * HIDDEN); deq(getT(gc, "blk.%d.attn_k.weight", il), HIDDEN_KV, HIDDEN, W[il].wk.data());
        W[il].wv.resize((size_t) HIDDEN_KV * HIDDEN); deq(getT(gc, "blk.%d.attn_v.weight", il), HIDDEN_KV, HIDDEN, W[il].wv.data());
        W[il].wo.resize((size_t) HIDDEN * HIDDEN);    deq(getT(gc, "blk.%d.attn_output.weight", il), HIDDEN, HIDDEN, W[il].wo.data());
        W[il].g.resize((size_t) INTER * HIDDEN);      deq(getT(gc, "blk.%d.ffn_gate.weight", il), INTER, HIDDEN, W[il].g.data());
        W[il].u.resize((size_t) INTER * HIDDEN);      deq(getT(gc, "blk.%d.ffn_up.weight", il),   INTER, HIDDEN, W[il].u.data());
        W[il].d.resize((size_t) HIDDEN * INTER);      deq(getT(gc, "blk.%d.ffn_down.weight", il), HIDDEN, INTER, W[il].d.data());
    }
    ggml_free(gc); gguf_free(gguf);

    // arena
    size_t bytes = (size_t) 2 * N_VOCAB * HIDDEN * 4 + (size_t) N_LAYERS * (2 * HIDDEN * HIDDEN + 2 * HIDDEN_KV * HIDDEN + 3 * INTER * HIDDEN) * 4;
    size_t ctx_size = bytes + bytes / 2 + ((size_t) 512 << 20);
    ggml_init_params params { ctx_size, nullptr, false };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, N_VOCAB); memcpy(embd->data, embd_d.data(), ggml_nbytes(embd));
    ggml_tensor * idt  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOKENS);        memcpy(idt->data, ids.data(), N_TOKENS * sizeof(int32_t));
    ggml_tensor * pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOKENS);        for (int i = 0; i < N_TOKENS; i++) ((int32_t *) pos->data)[i] = i;
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_TOKENS, N_TOKENS);
    { float * m = (float *) mask->data; for (int q = 0; q < N_TOKENS; q++) for (int k = 0; k < N_TOKENS; k++) m[q * N_TOKENS + k] = (k <= q) ? 0.f : -INFINITY; }
    ggml_tensor * onw  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN); memcpy(onw->data, outnorm_d.data(), ggml_nbytes(onw));
    ggml_tensor * oww  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, N_VOCAB); memcpy(oww->data, outw_d.data(), ggml_nbytes(oww));

    std::vector<LayerW> lw(N_LAYERS);
    auto mk1 = [&](std::vector<float> & s) { ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) s.size()); memcpy(t->data, s.data(), ggml_nbytes(t)); return t; };
    auto mk2 = [&](std::vector<float> & s, int ne0, int ne1) { ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1); memcpy(t->data, s.data(), ggml_nbytes(t)); return t; };
    for (int il = 0; il < N_LAYERS; il++) {
        lw[il].attn_norm = mk1(W[il].an);
        lw[il].wq = mk2(W[il].wq, HIDDEN, HIDDEN);
        lw[il].wk = mk2(W[il].wk, HIDDEN, HIDDEN_KV);
        lw[il].wv = mk2(W[il].wv, HIDDEN, HIDDEN_KV);
        lw[il].wo = mk2(W[il].wo, HIDDEN, HIDDEN);
        lw[il].ffn_norm = mk1(W[il].fn);
        lw[il].gate = mk2(W[il].g, HIDDEN, INTER);
        lw[il].up   = mk2(W[il].u, HIDDEN, INTER);
        lw[il].down = mk2(W[il].d, INTER, HIDDEN);
    }

    ggml_tensor * cur = ggml_get_rows(ctx, embd, idt);
    std::vector<ggml_tensor *> louts;
    for (int il = 0; il < N_LAYERS; il++) {
        cur = build_layer(ctx, cur, lw[il], pos, mask, HEAD_DIM, N_HEAD, N_HEAD_KV, N_TOKENS);
        louts.push_back(cur);
    }
    ggml_tensor * normed_final = ggml_mul(ctx, ggml_rms_norm(ctx, cur, REAL_RMS_EPS), onw);
    ggml_tensor * logits = ggml_mul_mat(ctx, oww, normed_final);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(gf, logits);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    fprintf(stderr, "=== recon CPU (build_layer) ===\n");
    argmax_cols((const float *) logits->data, N_VOCAB, N_TOKENS);
    for (int il = 0; il < N_LAYERS; il++) {
        const float * d = (const float *) louts[il]->data;   // [hidden, n_tok], last token = (n_tok-1)*hidden
        const float * lt = d + (size_t) (N_TOKENS - 1) * HIDDEN;
        fprintf(stderr, "l_out-%d last[0..3]= %.5f %.5f %.5f %.5f\n", il, lt[0], lt[1], lt[2], lt[3]);
    }
    return 0;
}
