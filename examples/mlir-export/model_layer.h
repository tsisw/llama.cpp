#pragma once
// Shared cache-free TinyLlama transformer layer (op set: rms_norm, mul, mul_mat, reshape, rope,
// permute, soft_max_ext, cont, silu, add). Used by live_graph_builder.h (reconstruction from
// llama.cpp's live cgraph) and recon_cpu_check.cpp (CPU check).
#include "ggml.h"

#include <cmath>

static float REAL_RMS_EPS = 9.999999747378752e-06f;  // overridden from the model (rms_epsilon)

// RoPE + attention-scale parameters. Defaults match ggml_rope(NORMAL); live_graph_builder.h
// overrides them from the live graph's ROPE / SOFT_MAX op_params so the reconstruction matches
// llama.cpp exactly. REAL_KQ_SCALE < 0 means "use 1/sqrt(head_dim)".
struct RopeParams {
    int   n_dims      = 0;   // 0 -> use head_dim
    int   mode        = 0;   // GGML_ROPE_TYPE_NORMAL
    int   n_ctx_orig  = 0;
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast   = 0.0f;
    float beta_slow   = 0.0f;
};
static RopeParams REAL_ROPE;
static float      REAL_KQ_SCALE = -1.0f;   // <0 -> 1/sqrt(head_dim)

struct LayerW {
    struct ggml_tensor * attn_norm, * wq, * wk, * wv, * wo, * ffn_norm, * gate, * up, * down;
};

static struct ggml_tensor * wg_rope(struct ggml_context * ctx, struct ggml_tensor * x,
                                     struct ggml_tensor * pos, int head_dim) {
    int nd = REAL_ROPE.n_dims > 0 ? REAL_ROPE.n_dims : head_dim;
    return ggml_rope_ext(ctx, x, pos, NULL, nd, REAL_ROPE.mode, REAL_ROPE.n_ctx_orig,
                         REAL_ROPE.freq_base, REAL_ROPE.freq_scale, REAL_ROPE.ext_factor,
                         REAL_ROPE.attn_factor, REAL_ROPE.beta_fast, REAL_ROPE.beta_slow);
}

static struct ggml_tensor * build_layer(struct ggml_context * ctx, struct ggml_tensor * x, const LayerW & lw,
                                         struct ggml_tensor * pos, struct ggml_tensor * mask, int head_dim,
                                         int n_head, int n_head_kv, int n_tokens) {
    int hidden = head_dim * n_head;

    struct ggml_tensor * normed1 = ggml_mul(ctx, ggml_rms_norm(ctx, x, REAL_RMS_EPS), lw.attn_norm);

    struct ggml_tensor * q_proj = ggml_mul_mat(ctx, lw.wq, normed1);
    struct ggml_tensor * k_proj = ggml_mul_mat(ctx, lw.wk, normed1);
    struct ggml_tensor * v_proj = ggml_mul_mat(ctx, lw.wv, normed1);

    struct ggml_tensor * q_heads = ggml_reshape_3d(ctx, q_proj, head_dim, n_head, n_tokens);
    struct ggml_tensor * k_heads = ggml_reshape_3d(ctx, k_proj, head_dim, n_head_kv, n_tokens);
    struct ggml_tensor * v_heads = ggml_reshape_3d(ctx, v_proj, head_dim, n_head_kv, n_tokens);

    struct ggml_tensor * q_rope = wg_rope(ctx, q_heads, pos, head_dim);
    struct ggml_tensor * k_rope = wg_rope(ctx, k_heads, pos, head_dim);

    struct ggml_tensor * q_perm = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
    struct ggml_tensor * k_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);

    struct ggml_tensor * kq = ggml_mul_mat(ctx, k_perm, q_perm);

    const float kq_scale = REAL_KQ_SCALE > 0.0f ? REAL_KQ_SCALE : 1.0f / sqrtf((float) head_dim);
    struct ggml_tensor * kq_soft = ggml_soft_max_ext(ctx, kq, mask, kq_scale, 0.0f);

    struct ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v_heads, 1, 2, 0, 3));

    struct ggml_tensor * kqv      = ggml_mul_mat(ctx, v_perm, kq_soft);
    struct ggml_tensor * kqv_perm = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    struct ggml_tensor * cur      = ggml_cont_2d(ctx, kqv_perm, hidden, n_tokens);

    struct ggml_tensor * attn_out = ggml_mul_mat(ctx, lw.wo, cur);
    struct ggml_tensor * resid1   = ggml_add(ctx, x, attn_out);

    struct ggml_tensor * normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, resid1, REAL_RMS_EPS), lw.ffn_norm);

    struct ggml_tensor * gate_o = ggml_mul_mat(ctx, lw.gate, normed2);
    struct ggml_tensor * act    = ggml_silu(ctx, gate_o);
    struct ggml_tensor * up_o   = ggml_mul_mat(ctx, lw.up, normed2);
    struct ggml_tensor * gated  = ggml_mul(ctx, act, up_o);
    struct ggml_tensor * down_o = ggml_mul_mat(ctx, lw.down, gated);

    return ggml_add(ctx, resid1, down_o);
}
