// Standalone repro: does ggml_add's broadcast chunking (nr0 = ne00/ne10 > 1)
// produce correct results on the Tsavorite backend when ne10*sizeof(float)
// is NOT a multiple of 128 bytes? The default width below (240) is chosen
// only because it is NOT a 32-float/128-byte multiple (240*4=960 bytes,
// 960 % 128 = 64, so successive chunks alternate 0/64-byte phase) -- it is
// NOT Gemma4's real head_dim. Gemma4-12b's actual head_dim is 256 (16 query
// heads, 8 grouped KV heads), already a clean 128-byte multiple; see
// GEMMA4-VALIDATION-SUMMARY.md for how that was confirmed by direct tracing.
//
// The existing simple-backend-tsi.cpp harness never sets nr0 > 1 -- its
// "scale" test uses equal-sized A/B (nr0 always 1), so it never exercises
// this alternating-phase multi-chunk broadcast path.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tsavorite.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level; (void) user_data;
    fputs(text, stderr);
}

static bool close_enough(float a, float b) {
    if (fabsf(a) < 1e-2f && fabsf(b) < 1e-2f) return fabsf(a - b) < 1e-6f;
    const float epsilon = 1e-3f;
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return diff < epsilon * max_val;
}

int main(int argc, char *argv[]) {
    const int64_t HEAD = (argc > 2) ? atoll(argv[2]) : 240;   // intentionally not a 32-float multiple; NOT Gemma4's real head_dim (see header comment above)
    const int64_t NCHUNKS = (argc > 1) ? atoll(argv[1]) : 4;
    if (HEAD <= 0 || NCHUNKS <= 0) {
        fprintf(stderr, "invalid dimensions: HEAD=%ld NCHUNKS=%ld (both must be > 0)\n", (long)HEAD, (long)NCHUNKS);
        return 1;
    }
    const int64_t NA = HEAD * NCHUNKS;        // ne00
    const int64_t NB = HEAD;                  // ne10 -- broadcasts across NCHUNKS
    const char *op = (argc > 3) ? argv[3] : "add";
    const bool is_rms = !strcmp(op, "rms_norm");
    const bool is_rms_looped = !strcmp(op, "rms_norm_looped");
    const bool is_known_op = is_rms || is_rms_looped ||
        !strcmp(op, "add") || !strcmp(op, "mul") || !strcmp(op, "sub") || !strcmp(op, "div");
    if (!is_known_op) {
        fprintf(stderr, "unrecognized op '%s' (expected add|mul|sub|div|rms_norm|rms_norm_looped)\n", op);
        return 1;
    }

    fprintf(stderr, "Repro: op=%s A=%ld elements, nr0-like=%ld chunks/rows of %ld floats (%ld bytes, mod128=%ld)\n",
            op, (long)NA, (long)NCHUNKS, (long)HEAD, (long)(HEAD*sizeof(float)), (long)((HEAD*sizeof(float)) % 128));

    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_t backend = ggml_backend_tsavorite_init();
    if (!backend) {
        fprintf(stderr, "ggml_backend_tsavorite_init() failed\n");
        return 1;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(2 + 2*NCHUNKS),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // For rms_norm: a is a 2D tensor [HEAD, NCHUNKS] -- NCHUNKS independent
    // rows of HEAD elements each, normalized per-row. This mimics QK-norm
    // (RMS_NORM applied per attention head, many heads/rows in one dispatch)
    // -- a pattern never exercised by simple-backend-tsi's rms_norm test,
    // which only ever tests a single 1D row.
    // For rms_norm_looped: NCHUNKS separate 1D tensors of HEAD elements each,
    // each its own ggml_rms_norm node -- tests whether looping single-row
    // dispatches (which always passes alone) avoids the multi-row bug above.
    struct ggml_tensor * a = (is_rms)
        ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD, NCHUNKS)
        : (is_rms_looped ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NA));
    struct ggml_tensor * b = (is_rms || is_rms_looped) ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NB);
    std::vector<struct ggml_tensor *> a_rows;
    if (is_rms_looped) {
        for (int64_t row = 0; row < NCHUNKS; row++) {
            a_rows.push_back(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD));
        }
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed\n");
        return 1;
    }

    std::vector<float> a_data(NA), b_data(NB), expected(NA);
    for (int64_t i = 0; i < NA; i++) a_data[i] = 0.01f * (float)(i + 1);
    for (int64_t i = 0; i < NB; i++) b_data[i] = 1.0f + 0.001f * (float)i;

    if (is_rms || is_rms_looped) {
        const float eps = 1e-5f;
        for (int64_t row = 0; row < NCHUNKS; row++) {
            double ss = 0.0;
            for (int64_t k = 0; k < HEAD; k++) {
                float v = a_data[row * HEAD + k];
                ss += (double)v * (double)v;
            }
            float scale = 1.0f / sqrtf((float)(ss / (double)HEAD) + eps);
            for (int64_t k = 0; k < HEAD; k++) {
                expected[row * HEAD + k] = a_data[row * HEAD + k] * scale;
            }
        }
    } else {
        for (int64_t i = 0; i < NA; i++) {
            float av = a_data[i], bv = b_data[i % HEAD];
            expected[i] = (!strcmp(op, "mul")) ? av * bv :
                          (!strcmp(op, "sub")) ? av - bv :
                          (!strcmp(op, "div")) ? av / bv :
                          av + bv;
        }
    }

    if (is_rms_looped) {
        for (int64_t row = 0; row < NCHUNKS; row++) {
            ggml_backend_tensor_set(a_rows[row], a_data.data() + row*HEAD, 0, ggml_nbytes(a_rows[row]));
        }
    } else {
        ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
        if (b) ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    }

    // rms_norm_looped builds one graph node per row (NCHUNKS of them), which
    // can exceed GGML_DEFAULT_GRAPH_SIZE (2048) for a large -c NCHUNKS -- size
    // the graph (and its backing context) from NCHUNKS instead of assuming
    // the default always fits.
    const size_t graph_size = is_rms_looped
        ? std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, (size_t)NCHUNKS + 8)
        : GGML_DEFAULT_GRAPH_SIZE;
    size_t buf_size = ggml_tensor_overhead()*graph_size + ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, graph_size, false);

    std::vector<struct ggml_tensor *> row_results;
    struct ggml_tensor * result = nullptr;
    if (is_rms_looped) {
        for (int64_t row = 0; row < NCHUNKS; row++) {
            struct ggml_tensor * r = ggml_rms_norm(ctx0, a_rows[row], 1e-5f);
            row_results.push_back(r);
            ggml_build_forward_expand(gf, r);
        }
    } else {
        result =
            is_rms ?                    ggml_rms_norm(ctx0, a, 1e-5f) :
            (!strcmp(op, "mul")) ? ggml_mul(ctx0, a, b) :
            (!strcmp(op, "sub")) ? ggml_sub(ctx0, a, b) :
            (!strcmp(op, "div")) ? ggml_div(ctx0, a, b) :
            ggml_add(ctx0, a, b);
        ggml_build_forward_expand(gf, result);
    }

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute failed, status=%d\n", (int)st);
        ggml_gallocr_free(allocr);
        ggml_free(ctx0);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> out_data(NA);
    if (is_rms_looped) {
        for (int64_t row = 0; row < NCHUNKS; row++) {
            ggml_backend_tensor_get(row_results[row], out_data.data() + row*HEAD, 0, ggml_nbytes(row_results[row]));
        }
    } else {
        ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));
    }

    int fail_count = 0;
    int64_t first_fail = -1;
    for (int64_t i = 0; i < NA; i++) {
        if (!close_enough(out_data[i], expected[i])) {
            fail_count++;
            if (first_fail < 0) first_fail = i;
        }
    }

    if (fail_count == 0) {
        fprintf(stderr, "\nTEST CASE PASSED (all %ld elements correct)\n", (long)NA);
    } else {
        fprintf(stderr, "\nTEST CASE FAILED: %d/%ld elements wrong. First fail at i=%ld (chunk r=%ld, phase=%ld bytes): got %.6f expected %.6f\n",
                fail_count, (long)NA, (long)first_fail, (long)(first_fail / HEAD),
                (long)((first_fail / HEAD) * HEAD * sizeof(float)) % 128,
                out_data[first_fail], expected[first_fail]);
        // dump a window around the first failure
        int64_t lo = std::max<int64_t>(0, first_fail - 4);
        int64_t hi = std::min<int64_t>(NA, first_fail + 8);
        for (int64_t i = lo; i < hi; i++) {
            fprintf(stderr, "  i=%ld got=%.6f expected=%.6f\n", (long)i, out_data[i], expected[i]);
        }
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return fail_count == 0 ? 0 : 1;
}
