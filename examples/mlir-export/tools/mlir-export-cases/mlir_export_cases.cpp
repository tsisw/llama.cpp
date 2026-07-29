// Emits self-contained test cases for the ggml -> linalg MLIR exporter.
//
// Per case: build a small ggml graph, fill its inputs from a fixed seed, compute the CPU reference
// with ggml_graph_compute_with_ctx, export the graph via exporter.h, and write everything to a case
// directory that tests/test_mlir_export.py can compile and check without touching ggml.
//
// Links ggml only (never llama) - see the note in CMakeLists.txt.
//
//   mlir-export-cases --list
//   mlir-export-cases --emit <name> <dir>
//   mlir-export-cases --emit-all <dir>
#include "tsi/export/TextEmitter.h"

#if TSI_HAVE_MLIR_EXPORT
#    include "tsi/export/Exporter.h"
#endif

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

// mt19937 is standard-specified, so this is reproducible across stdlib implementations;
// std::uniform_real_distribution is not. Values land in [-1, 1].
static void fill_seeded(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    float *      d = (float *) t->data;
    const size_t n = ggml_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = ((float) (rng() % 20001) - 10000.0f) / 10000.0f;
    }
}

// I32 index inputs (GET_ROWS token ids) must stay inside the table, and position inputs (ROPE)
// must be a plausible sequence - neither is expressible as "random in [-1,1]". Only the builder
// knows the valid range, so builders fill their own I32 tensors and emit_case skips them.
static void fill_seeded_i32(ggml_tensor * t, uint32_t seed, int32_t hi) {
    std::mt19937 rng(seed);
    int32_t *    d = (int32_t *) t->data;
    const size_t n = ggml_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = (int32_t) (rng() % (uint32_t) hi);
    }
}

// Sequential positions from `start`. Starting above 0 matters: at pos 0 every RoPE angle is 0 and
// the rotation degenerates to the identity, which would pass even a completely broken emitter.
static void fill_positions(ggml_tensor * t, int32_t start) {
    int32_t *    d = (int32_t *) t->data;
    const size_t n = ggml_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = start + (int32_t) i;
    }
}

static const char * dtype_of(const ggml_tensor * t) {
    return t->type == GGML_TYPE_I32 ? "i32" : "f32";
}

// MLIR shape = ne reversed over n_dims (exporter.h mlir_shape_dims).
static std::vector<int64_t> mlir_shape_of(const ggml_tensor * t) {
    std::vector<int64_t> s;
    for (int i = ggml_n_dims(t) - 1; i >= 0; i--) {
        s.push_back(t->ne[i]);
    }
    return s;
}

// Raw dump of t's buffer. ggml_nbytes (not nelements*4) so I32 inputs work too; every tensor
// written here is contiguous, which is why the view-producing cases below wrap in ggml_cont.
static void write_tensor(const fs::path & p, const ggml_tensor * t) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char *) t->data, (std::streamsize) ggml_nbytes(t));
}

static std::string shape_json(const std::vector<int64_t> & s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); i++) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

// ---------------------------------------------------------------------------------------
// case definitions
// ---------------------------------------------------------------------------------------

// Builds the graph, appends every func-argument leaf to `args` in %arg order, returns the output.
using build_fn = ggml_tensor * (*) (ggml_context * ctx, std::vector<const ggml_tensor *> & args);

struct case_spec {
    const char * name;
    build_fn     build;
    float        rtol;
    float        atol;
    const char * expect;   // "pass" | "mismatch"
    bool         corrupt;  // deliberately poison expected_0.bin (harness self-check)
};

static ggml_tensor * build_add(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_add(ctx, a, b);
}

static ggml_tensor * build_mul(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul(ctx, a, b);
}

static ggml_tensor * build_scale(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_scale(ctx, a, 0.5f);   // scalar is baked into the graph, not a func arg
}

static ggml_tensor * build_silu(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_silu(ctx, a);          // GGML_OP_UNARY / GGML_UNARY_OP_SILU
}

// RMS_NORM normalizes over ne[0], so use 2-D input to exercise a real reduction per row.
static ggml_tensor * build_rms_norm(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_rms_norm(ctx, a, 1e-5f);
}

static ggml_tensor * build_soft_max(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_soft_max(ctx, a);
}

// ggml_mul_mat(a,b) requires a->ne[0] == b->ne[0] (= K) and yields ne = (a->ne[1], b->ne[1]).
// In MLIR shape order that is a -> [M,K], b -> [N,K], out -> [N,M], computed as B @ A^T (see
// emit_mul_mat_2d in exporter.h). K is a multiple of 32 for TMU K-alignment (TMU_K_MULTIPLE).
static ggml_tensor * build_matmul(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int K = 32, M = 32, N = 32;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);   // MLIR [M,K]
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);   // MLIR [N,K]
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul_mat(ctx, a, b);                                  // ne (M,N) -> MLIR [N,M]
}

static ggml_tensor * build_matmul_add(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int K = 32, M = 32, N = 32;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);   // matches mul_mat's ne
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");
    args.push_back(a);
    args.push_back(b);
    args.push_back(c);
    return ggml_add(ctx, ggml_mul_mat(ctx, a, b), c);
}

// ---------------------------------------------------------------------------------------
// matmul rank variants
// ---------------------------------------------------------------------------------------

// b rank-1 (the n_tokens=1 decode shape): ne = (M,1,1,1) collapses to rank 1, so this reaches
// emit_mul_mat_2d_vec rather than the plain 2D path.
static ggml_tensor * build_matmul_vec(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     K = 32, M = 32;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul_mat(ctx, a, b);
}

// Equal head counts on both operands: emit_mul_mat_batched_3d, no broadcast.
static ggml_tensor * build_matmul_3d(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     K = 32, M = 32, N = 32, H = 2;
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, H);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, H);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul_mat(ctx, a, b);
}

// b carries 2x a's heads (b->ne[2] % a->ne[2] == 0, unequal): the real Q/KV head mismatch, so this
// reaches emit_mul_mat_batched_3d_gqa and through it emit_repeat_heads_3d.
static ggml_tensor * build_matmul_gqa(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     K = 32, M = 32, N = 32;
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, 2);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, 4);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul_mat(ctx, a, b);
}

// ---------------------------------------------------------------------------------------
// shape ops
// ---------------------------------------------------------------------------------------
//
// PERMUTE and RESHAPE return ggml *views*: the result aliases its source buffer with reordered
// strides, so dumping it linearly would not reflect the permutation. ggml_cont materializes a
// contiguous copy, which is what write_tensor needs. The CONT node itself is a same-shape
// passthrough in the exporter, so it costs nothing in the emitted IR.

// Rank-3 permute that genuinely moves data: axis {1,0,2,3} swaps two non-1 dims, so the
// size1-only fast path does not trigger and a real linalg.transpose is emitted.
static ggml_tensor * build_permute(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 2);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_cont(ctx, ggml_permute(ctx, a, 1, 0, 2, 3));
}

// Permute that only reshuffles size-1 dims (ne[1] == 1), so the relative order of the non-1 dims
// is preserved, no data moves, and emit_permute delegates to emit_size1_reshape.
static ggml_tensor * build_permute_size1(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 1, 4);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
}

// 2D -> 3D head split: node->ne[2] == x->ne[1] and ne[0]*ne[1] == x->ne[0], the shape real
// attention uses before per-head ops. tensor.expand_shape.
static ggml_tensor * build_reshape_split(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);   // (hidden, n_tokens)
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_cont(ctx, ggml_reshape_3d(ctx, a, 16, 4, 4));          // (head_dim, n_head, n_tokens)
}

// 3D -> 2D head merge, the inverse of the above. tensor.collapse_shape.
static ggml_tensor * build_reshape_merge(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 16, 4, 4);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_cont(ctx, ggml_reshape_2d(ctx, a, 64, 4));
}

// ggml_concat along ggml dim 1: the KV-cache append pattern, built from empty + insert_slice
// because tensor.concat is not bufferized by the TSI pipeline.
static ggml_tensor * build_concat(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 8, 2);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 4, 2);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_concat(ctx, a, b, 1);
}

// Token embedding lookup. The ids are genuine runtime data, so this is the first case with a
// non-f32 function argument, and each gathered row needs a dynamic-offset extract_slice.
static ggml_tensor * build_get_rows(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     n_embd = 32, n_vocab = 16, n_tokens = 4;
    ggml_tensor * tbl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(tbl, "tbl");
    ggml_set_name(ids, "ids");
    fill_seeded_i32(ids, 0x5EEDu, n_vocab);
    args.push_back(tbl);
    args.push_back(ids);
    return ggml_get_rows(ctx, tbl, ids);
}

// n_tokens == 1: the output collapses to rank 1, which takes emit_get_rows' rank-reducing branch
// instead of the unrolled extract/insert loop.
static ggml_tensor * build_get_rows_1tok(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     n_embd = 32, n_vocab = 16;
    ggml_tensor * tbl = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(tbl, "tbl");
    ggml_set_name(ids, "ids");
    fill_seeded_i32(ids, 0x5EEEu, n_vocab);
    args.push_back(tbl);
    args.push_back(ids);
    return ggml_get_rows(ctx, tbl, ids);
}

// ---------------------------------------------------------------------------------------
// rope
// ---------------------------------------------------------------------------------------
// Both require n_dims == head_dim (full-row rotation), GGML_ROPE_TYPE_NORMAL, and ggml_rope's
// default freq params (freq_scale=1, ext_factor=0, attn_factor=1) - the exporter rejects anything
// else.

// rank-2 x: ne[2] is implicitly 1, so pos holds exactly one shared position for every row.
static ggml_tensor * build_rope_2d(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     head_dim = 16, n_head = 4;
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, n_head);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(x, "x");
    ggml_set_name(pos, "pos");
    fill_positions(pos, 5);
    args.push_back(x);
    args.push_back(pos);
    return ggml_rope(ctx, x, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
}

// rank-3 x: one position per token (pos->ne[0] == x->ne[2]), broadcast across n_head.
static ggml_tensor * build_rope_3d(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int     head_dim = 16, n_head = 4, n_tokens = 3;
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(x, "x");
    ggml_set_name(pos, "pos");
    fill_positions(pos, 3);
    args.push_back(x);
    args.push_back(pos);
    return ggml_rope(ctx, x, pos, head_dim, GGML_ROPE_TYPE_NORMAL);
}

static const case_spec CASES[] = {
    { "add",          build_add, 0.0f, 0.0f, "pass",     false },
    // Proves the comparison in test_mlir_export.py actually compares. If a harness bug made the
    // check vacuous, every other case would still pass and this one would too - so this must fail
    // to match, by construction.
    { "add_negative", build_add, 0.0f, 0.0f, "mismatch", true  },
    { "mul",          build_mul,      0.0f,  0.0f,  "pass", false },
    { "scale",        build_scale,    0.0f,  0.0f,  "pass", false },
    { "silu",         build_silu,     1e-5f, 1e-6f, "pass", false },
    { "rms_norm",     build_rms_norm, 1e-5f, 1e-6f, "pass", false },
    { "soft_max",     build_soft_max, 1e-5f, 1e-6f, "pass", false },
    // atol 1e-5, not 1e-6: a 32x32x32 f32 matmul measures max abs err ~5.7e-06 from reduction
    // reassociation in the lowered code. Measured, not guessed.
    { "matmul",       build_matmul,     1e-5f, 1e-5f, "pass", false },
    { "matmul_add",   build_matmul_add, 1e-5f, 1e-5f, "pass", false },

    // Same reduction-reassociation tolerance as matmul, for the same reason. Measured max abs err
    // on FFM: vec 2.4e-07, 3d 5.1e-07, gqa 9.5e-07 - all inside atol 1e-5 with ~10x headroom.
    { "matmul_vec",   build_matmul_vec, 1e-5f, 1e-5f, "pass", false },
    { "matmul_3d",    build_matmul_3d,  1e-5f, 1e-5f, "pass", false },
    { "matmul_gqa",   build_matmul_gqa, 1e-5f, 1e-5f, "pass", false },

    // Pure data movement, no arithmetic, so these are held to BIT-EXACT equality (rtol=atol=0).
    // All seven measure max abs err exactly 0.0, so the zero tolerance is a real constraint and
    // not an accident waiting to flake.
    { "permute",        build_permute,        0.0f, 0.0f, "pass", false },
    { "permute_size1",  build_permute_size1,  0.0f, 0.0f, "pass", false },
    { "reshape_split",  build_reshape_split,  0.0f, 0.0f, "pass", false },
    { "reshape_merge",  build_reshape_merge,  0.0f, 0.0f, "pass", false },
    { "concat",         build_concat,         0.0f, 0.0f, "pass", false },
    { "get_rows",       build_get_rows,       0.0f, 0.0f, "pass", false },
    { "get_rows_1tok",  build_get_rows_1tok,  0.0f, 0.0f, "pass", false },

    // RoPE recomputes cos/sin in the emitted IR rather than reusing ggml's, so it cannot be exact.
    // Measured max abs err on FFM: 2d 1.5e-07, 3d 1.8e-07, both well inside atol 1e-6.
    { "rope_2d",      build_rope_2d,    1e-5f, 1e-6f, "pass", false },
    { "rope_3d",      build_rope_3d,    1e-5f, 1e-6f, "pass", false },
};

static const size_t N_CASES = sizeof(CASES) / sizeof(CASES[0]);

// ---------------------------------------------------------------------------------------
// emit
// ---------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------
// emitter selection (TRANSITIONAL)
// ---------------------------------------------------------------------------------------
// The string emitter (TextEmitter.h) is being replaced by the MLIR C++ API one (Exporter.h).
// While the port is in progress both are reachable, so each ported family can be diffed against
// the committed golden IR without leaving the default emitter broken. Once every emitter is
// ported this flag, the text emitter, and this shim all go away.
enum class emitter_kind { text, mlir };

static emitter_kind g_emitter = emitter_kind::text;

static std::string emit_forward_mlir(ggml_cgraph * gf, const std::vector<const ggml_tensor *> & args) {
    if (g_emitter == emitter_kind::text) {
        return "module {\n" + build_func_text_baked(gf, "forward", args, {}) + "}\n";
    }
#if TSI_HAVE_MLIR_EXPORT
    tsi::mlir_export::ExportOptions opts;
    opts.runtime_args = args;
    try {
        return tsi::mlir_export::exportGraph(gf, opts);
    } catch (const tsi::mlir_export::mlir_export_error & e) {
        // Re-thrown as the text emitter's type so emit_case has one exception type to handle for
        // as long as both emitters coexist.
        throw mlir_export_error(e.what());
    }
#else
    fprintf(stderr, "--emitter mlir needs the tsi-mlir-export library; configure with MLIR available\n");
    exit(2);
#endif
}

static bool emit_case(const case_spec & spec, const fs::path & dir) {
    fs::create_directories(dir);

    ggml_init_params ip { (size_t) 256 << 20, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", spec.name);
        return false;
    }

    std::vector<const ggml_tensor *> args;
    ggml_tensor *                    out = spec.build(ctx, args);

    // Seed per argument index, offset by a per-case hash so different cases get different data.
    uint32_t base = 0x9E3779B9u;
    for (const char * p = spec.name; *p; p++) base = base * 31u + (uint32_t) *p;
    for (size_t i = 0; i < args.size(); i++) {
        // I32 args (GET_ROWS ids, ROPE positions) are filled by the builder, which is the only
        // place that knows their valid range. Overwriting them with random floats here would
        // produce out-of-range indices and a garbage reference.
        if (args[i]->type == GGML_TYPE_F32) {
            fill_seeded(const_cast<ggml_tensor *>(args[i]), base + (uint32_t) i);
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_graph_compute_with_ctx failed\n", spec.name);
        ggml_free(ctx);
        return false;
    }

    std::string expect = spec.expect;
    std::string mlir;
    try {
        mlir = emit_forward_mlir(gf, args);
    } catch (const mlir_export_error & e) {
        // Exporter gap: record it so the runner xfails with a reason instead of the build breaking.
        fprintf(stderr, "%s: exporter rejected the graph: %s\n", spec.name, e.what());
        expect = "unsupported";
        mlir   = "";
    }

    std::ofstream(dir / "forward.mlir") << mlir;

    std::string args_json;
    for (size_t i = 0; i < args.size(); i++) {
        std::string fn = "input_" + std::to_string(i) + ".bin";
        write_tensor(dir / fn, args[i]);
        if (i) args_json += ",\n             ";
        args_json += "{\"file\": \"" + fn + "\", \"shape\": " + shape_json(mlir_shape_of(args[i])) +
                     ", \"dtype\": \"" + dtype_of(args[i]) + "\"}";
    }

    if (spec.corrupt) {
        // Offset element 0 by a large, unmistakable amount, then write.
        std::vector<float> ref(ggml_nelements(out));
        memcpy(ref.data(), out->data, ref.size() * sizeof(float));
        ref[0] += 1000.0f;
        std::ofstream f(dir / "expected_0.bin", std::ios::binary);
        f.write((const char *) ref.data(), (std::streamsize) (ref.size() * sizeof(float)));
    } else {
        write_tensor(dir / "expected_0.bin", out);
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%.8g", spec.rtol);
    std::string rtol = buf;
    snprintf(buf, sizeof(buf), "%.8g", spec.atol);
    std::string atol = buf;

    std::ofstream(dir / "case.json")
        << "{\n"
        << "  \"name\": \"" << spec.name << "\",\n"
        << "  \"expect\": \"" << expect << "\",\n"
        << "  \"rtol\": " << rtol << ",\n"
        << "  \"atol\": " << atol << ",\n"
        << "  \"args\": [" << args_json << "],\n"
        << "  \"output\": {\"file\": \"expected_0.bin\", \"shape\": "
        << shape_json(mlir_shape_of(out)) << "}\n"
        << "}\n";

    ggml_free(ctx);
    printf("emitted %s -> %s\n", spec.name, dir.c_str());
    return true;
}

int main(int argc, char ** argv) {
    // Optional leading "--emitter text|mlir", stripped before the normal argument handling below.
    if (argc >= 3 && strcmp(argv[1], "--emitter") == 0) {
        if (strcmp(argv[2], "mlir") == 0) {
            g_emitter = emitter_kind::mlir;
        } else if (strcmp(argv[2], "text") != 0) {
            fprintf(stderr, "unknown --emitter '%s' (expected text or mlir)\n", argv[2]);
            return 2;
        }
        argv += 2;
        argc -= 2;
    }

    if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
        for (size_t i = 0; i < N_CASES; i++) printf("%s\n", CASES[i].name);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "--emit") == 0) {
        for (size_t i = 0; i < N_CASES; i++) {
            if (strcmp(CASES[i].name, argv[2]) == 0) {
                return emit_case(CASES[i], argv[3]) ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown case: %s\n", argv[2]);
        return 1;
    }
    if (argc == 3 && strcmp(argv[1], "--emit-all") == 0) {
        for (size_t i = 0; i < N_CASES; i++) {
            if (!emit_case(CASES[i], fs::path(argv[2]) / CASES[i].name)) return 1;
        }
        return 0;
    }
    fprintf(stderr,
            "usage: %s --list\n"
            "       %s --emit <name> <dir>\n"
            "       %s --emit-all <dir>\n",
            argv[0], argv[0], argv[0]);
    return 1;
}
