// Does ggml_mul(a, b) with a=[3840,N,1,1] and b=[1,1,1,1] (Gemma4's exact
// out_scale broadcast: N=7 tokens, scalar per-layer weight) compute
// correctly? Earlier chunked-repro testing only ever used a 1D tensor (N=1
// implicit), covering the inner chunk loop (nr0=3840) but never the OUTER
// row loop (nr=N) that real prefill (N>1 tokens) actually exercises on top
// of it -- this tests both loops nested together, matching the real shape.
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
    const float epsilon = 1e-3f;
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return diff < epsilon * fmaxf(max_val, 1.0f);
}

int main(int argc, char *argv[]) {
    const int64_t D = (argc > 1) ? atoll(argv[1]) : 3840;
    const int64_t N = (argc > 2) ? atoll(argv[2]) : 7;

    fprintf(stderr, "2D scalar-broadcast repro: a=[%ld,%ld,1,1] b=[1,1,1,1] (ggml_mul)\n", (long)D, (long)N);

    ggml_log_set(ggml_log_callback_default, nullptr);

    ggml_backend_t backend = ggml_backend_tsavorite_init();
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 1; }

    struct ggml_init_params params {
        /*.mem_size=*/ ggml_tensor_overhead() * 4, /*.mem_buffer=*/ NULL, /*.no_alloc=*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { fprintf(stderr, "alloc_ctx_tensors failed\n"); return 1; }

    std::vector<float> a_data((size_t)D * (size_t)N);
    // Use real-scale magnitudes like the actual model's activations (up to ~300+), not tiny synthetic values.
    for (int64_t i = 0; i < D*N; i++) a_data[i] = 10.0f * sinf(0.01f * (float)(i + 1)) + ((i % 97 == 0) ? 300.0f : 0.0f);
    float b_val = 0.052979f; // matches the real blk.0.layer_output_scale.weight value observed
    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, &b_val, 0, sizeof(float));

    std::vector<float> expected((size_t)D*N);
    for (int64_t i = 0; i < D*N; i++) expected[i] = a_data[i] * b_val;

    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params0 = { /*.mem_size=*/ buf_size, /*.mem_buffer=*/ buf.data(), /*.no_alloc=*/ true };
    struct ggml_context * ctx0 = ggml_init(params0);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
    struct ggml_tensor * result = ggml_mul(ctx0, a, b);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(allocr, gf);
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph_compute failed status=%d\n", (int)st); return 1; }

    std::vector<float> out_data((size_t)D*N);
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    int fail_count = 0;
    int64_t first_fail = -1;
    for (size_t i = 0; i < out_data.size(); i++) {
        if (!close_enough(out_data[i], expected[i])) {
            fail_count++;
            if (first_fail < 0) first_fail = (int64_t)i;
        }
    }

    if (fail_count == 0) {
        fprintf(stderr, "\nTEST CASE PASSED (all %zu elements correct)\n", out_data.size());
    } else {
        fprintf(stderr, "\nTEST CASE FAILED: %d/%zu elements wrong. First fail at i=%ld (row=%ld): got=%.6f expected=%.6f\n",
                fail_count, out_data.size(), (long)first_fail, (long)(first_fail / D), out_data[first_fail], expected[first_fail]);
        int64_t lo = std::max<int64_t>(0, first_fail - 3);
        int64_t hi = std::min<int64_t>((int64_t)out_data.size(), first_fail + 6);
        for (int64_t i = lo; i < hi; i++) {
            fprintf(stderr, "  i=%ld row=%ld got=%.6f expected=%.6f\n", (long)i, (long)(i/D), out_data[i], expected[i]);
        }
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return fail_count == 0 ? 0 : 1;
}
