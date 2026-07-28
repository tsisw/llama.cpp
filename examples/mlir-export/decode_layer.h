#pragma once
// Fixed-max-length (L) KV-cache decode layer, shared by the CPU validator (decode_cpu_check) and the
// graph emitter so both drive the same graph. Processes one new token; cache_K/cache_V are
// [head_dim, n_head_kv, L] (L = cache_K->ne[2]); a runtime mask over the L+1 keys (cache ++ new)
// blocks the not-yet-filled cache slots.
#include "model_layer.h"   // REAL_RMS_EPS, REAL_ROPE, REAL_KQ_SCALE, wg_rope, LayerW
#include "ggml.h"

#include <cmath>

// Returns x_next; sets *k_new (roped K [head_dim,n_head_kv,1]) and *v_new to store at the cache slot.
static struct ggml_tensor * build_decode_layer(struct ggml_context * ctx, struct ggml_tensor * x,
        const LayerW & lw, struct ggml_tensor * pos,
        struct ggml_tensor * cache_K, struct ggml_tensor * cache_V, struct ggml_tensor * mask,
        int head_dim, int n_head, int n_head_kv,
        struct ggml_tensor ** k_new, struct ggml_tensor ** v_new) {
    const int hidden = head_dim * n_head;

    struct ggml_tensor * n1 = ggml_mul(ctx, ggml_rms_norm(ctx, x, REAL_RMS_EPS), lw.attn_norm);
    struct ggml_tensor * q  = ggml_mul_mat(ctx, lw.wq, n1);
    struct ggml_tensor * k  = ggml_mul_mat(ctx, lw.wk, n1);
    struct ggml_tensor * v  = ggml_mul_mat(ctx, lw.wv, n1);

    struct ggml_tensor * qh = ggml_reshape_3d(ctx, q, head_dim, n_head,    1);
    struct ggml_tensor * kh = ggml_reshape_3d(ctx, k, head_dim, n_head_kv, 1);
    struct ggml_tensor * vh = ggml_reshape_3d(ctx, v, head_dim, n_head_kv, 1);

    struct ggml_tensor * qr = wg_rope(ctx, qh, pos, head_dim);   // [head_dim, n_head,    1]
    struct ggml_tensor * kr = wg_rope(ctx, kh, pos, head_dim);   // [head_dim, n_head_kv, 1]  (-> cache)
    *k_new = kr;
    *v_new = vh;

    struct ggml_tensor * Kf = ggml_concat(ctx, cache_K, kr, 2);   // [head_dim, n_head_kv, L+1]
    struct ggml_tensor * Vf = ggml_concat(ctx, cache_V, vh, 2);

    struct ggml_tensor * qp = ggml_permute(ctx, qr, 0, 2, 1, 3);   // [head_dim, 1,   n_head]
    struct ggml_tensor * Kp = ggml_permute(ctx, Kf, 0, 2, 1, 3);   // [head_dim, L+1, n_head_kv]
    struct ggml_tensor * kq = ggml_mul_mat(ctx, Kp, qp);           // [L+1, 1, n_head]  (GQA broadcast)

    const float scale = REAL_KQ_SCALE > 0.0f ? REAL_KQ_SCALE : 1.0f / sqrtf((float) head_dim);
    struct ggml_tensor * soft = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);   // mask blocks unfilled slots

    struct ggml_tensor * Vp  = ggml_cont(ctx, ggml_permute(ctx, Vf, 1, 2, 0, 3)); // [L+1, head_dim, n_head_kv]
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, Vp, soft);                        // [head_dim, 1, n_head]
    struct ggml_tensor * cc  = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), hidden, 1);

    struct ggml_tensor * attn   = ggml_mul_mat(ctx, lw.wo, cc);
    struct ggml_tensor * resid1 = ggml_add(ctx, x, attn);
    struct ggml_tensor * n2     = ggml_mul(ctx, ggml_rms_norm(ctx, resid1, REAL_RMS_EPS), lw.ffn_norm);
    struct ggml_tensor * gate   = ggml_silu(ctx, ggml_mul_mat(ctx, lw.gate, n2));
    struct ggml_tensor * up     = ggml_mul_mat(ctx, lw.up, n2);
    struct ggml_tensor * down   = ggml_mul_mat(ctx, lw.down, ggml_mul(ctx, gate, up));
    return ggml_add(ctx, resid1, down);
}
