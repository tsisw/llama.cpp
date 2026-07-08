#ifndef MAT_MUL_TSI_TEST_CASE_CPP
#define MAT_MUL_TSI_TEST_CASE_CPP

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tsavorite.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/*
 * -----------------------------------------------------------------------------
 * MAT_MUL GGML / Tsavorite Backend Test Case
 * -----------------------------------------------------------------------------
 *
 * Purpose:
 *   This test validates the GGML MAT_MUL execution path using the Tsavorite
 *   backend. It is intended to exercise the same high-level GGML API flow used
 *   by model execution:
 *
 *       A = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, M, 1, 1)
 *       B = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, N, 1, 1)
 *       C = ggml_mul_mat(ctx, A, B)
 *       ggml_build_forward_expand(gf, C)
 *       ggml_backend_graph_compute(backend, gf)
 *
 * GGML MAT_MUL Shape Contract:
 *   GGML represents MUL_MAT inputs as:
 *
 *       A / src0 : [K, M, d2, d3]
 *       B / src1 : [K, N, d2, d3]
 *       C / dst  : [M, N, d2, d3]
 *
 *   This test currently validates the simple 2D case only:
 *
 *       A = [K, M, 1, 1]
 *       B = [K, N, 1, 1]
 *       C = [M, N, 1, 1]
 *
 * Memory Layout:
 *   GGML stores tensors with ne[0] as the fastest-changing dimension.
 *
 *   A memory index:
 *
 *       A[k + m*K]
 *
 *   B memory index:
 *
 *       B[k + n*K]
 *
 *   C memory index:
 *
 *       C[m + n*M]
 *
 * Reference Computation:
 *   The CPU reference result is computed as:
 *
 *       C[m, n] = sum over k:
 *
 *           A[k, m] * B[k, n]
 *
 *   This matches GGML's MUL_MAT interpretation:
 *
 *       C = transpose(A) * B
 *
 * Supported Command Lines:
 *
 *   Default shape:
 *
 *       ./simple-backend-tsi mat-mul
 *
 *   Explicit K/M/N:
 *
 *       ./simple-backend-tsi mat-mul 256 8 64
 *
 *   Explicit GGML-style 4D shapes:
 *
 *       ./simple-backend-tsi mat-mul [256 8 1 1] [256 64 1 1]
 *
 * Useful Test Shapes:
 *
 *   Basic Triton-aligned shape:
 *
 *       ./simple-backend-tsi mat-mul 256 8 64
 *
 *   Model-like shapes:
 *
 *       ./simple-backend-tsi mat-mul 576 5 576
 *       ./simple-backend-tsi mat-mul 1536 5 1536
 *       ./simple-backend-tsi mat-mul 2048 6 2048
 *
 * Current Scope:
 *   This test intentionally focuses on plain 2D MAT_MUL only.
 *
 *   It does NOT currently test:
 *
 *       - batched 3D / 4D MAT_MUL
 *       - d2 / d3 broadcast behavior
 *       - repeat mapping
 *       - quantized types
 *       - F16/BF16
 *       - non-F32 inputs
 *
 * Why This Test Exists:
 *   This test is useful for debugging differences between:
 *
 *       - generated host_wrapper MAT_MUL path
 *       - manual GGML-side pack-args MAT_MUL path
 *       - Triton MAT_MUL backend dispatch
 *
 *   It confirms that GGML creates the correct MAT_MUL graph and that the backend
 *   sees the expected tensor shapes before entering the Tsavorite MAT_MUL path.
 *
 * Expected Result:
 *   The output from the backend is copied back and compared against a CPU
 *   reference implementation. The test prints:
 *
 *       MAT_MUL TEST PASSED
 *
 *   if all output elements match within tolerance.
 * -----------------------------------------------------------------------------
 */

static bool matmul_tsi_float_equal(float a, float b) {
    if (fabsf(a) < 1e-2f && fabsf(b) < 1e-2f) {
        return fabsf(a - b) < 1e-5f;
    }

    const float eps  = 1e-3f;
    const float diff = fabsf(a - b);
    const float maxv = fmaxf(fabsf(a), fabsf(b));
    return diff <= eps * maxv;
}

static int matmul_tsi_parse_int_token(const char *s, bool *ok) {
    if (!s) {
        *ok = false;
        return 0;
    }

    std::string cleaned;
    for (const char *p = s; *p; ++p) {
        if ((*p >= '0' && *p <= '9') || *p == '-') {
            cleaned.push_back(*p);
        }
    }

    if (cleaned.empty() || cleaned == "-") {
        *ok = false;
        return 0;
    }

    *ok = true;
    return std::atoi(cleaned.c_str());
}

static void matmul_tsi_parse_shape_args(int argc, char **argv, int *K, int *M, int *N) {
    // Default requested by test case:
    // A=[256,8,1,1], B=[256,64,1,1], C=[8,64,1,1]
    *K = 256;
    *M = 8;
    *N = 64;

    std::vector<int> nums;
    for (int i = 2; i < argc; ++i) {
        bool ok = false;
        int v = matmul_tsi_parse_int_token(argv[i], &ok);
        if (ok) {
            nums.push_back(v);
        }
    }

    // Supported:
    //   ./simple-backend-tsi mat-mul 256 8 64
    if (nums.size() == 3) {
        *K = nums[0];
        *M = nums[1];
        *N = nums[2];
        return;
    }

    // Supported:
    //   ./simple-backend-tsi mat-mul [256 8 1 1] [256 64 1 1]
    if (nums.size() >= 8) {
        const int a0 = nums[0];
        const int a1 = nums[1];
        const int a2 = nums[2];
        const int a3 = nums[3];

        const int b0 = nums[4];
        const int b1 = nums[5];
        const int b2 = nums[6];
        const int b3 = nums[7];

        if (a0 != b0 || a2 != 1 || a3 != 1 || b2 != 1 || b3 != 1) {
            fprintf(stderr,
                    "ERROR: expected A=[K M 1 1], B=[K N 1 1], got "
                    "A=[%d %d %d %d], B=[%d %d %d %d]\n",
                    a0, a1, a2, a3, b0, b1, b2, b3);
            return;
        }

        *K = a0;
        *M = a1;
        *N = b1;
        return;
    }
}

static void matmul_tsi_fill_inputs(std::vector<float> &A,
                                   std::vector<float> &B,
                                   int K,
                                   int M,
                                   int N) {
    // GGML source A shape: [K, M, 1, 1]
    // Memory index: k + m*K
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            A[k + m*K] = 0.001f * (float)(k + 1) + 0.01f * (float)(m + 1);
        }
    }

    // GGML source B shape: [K, N, 1, 1]
    // Memory index: k + n*K
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            B[k + n*K] = 0.002f * (float)(k + 1) - 0.003f * (float)(n + 1);
        }
    }
}

static void matmul_tsi_reference(const std::vector<float> &A,
                                 const std::vector<float> &B,
                                 std::vector<float> &C,
                                 int K,
                                 int M,
                                 int N) {
    // GGML MUL_MAT contract:
    //   A/src0 = [K, M, 1, 1]
    //   B/src1 = [K, N, 1, 1]
    //   C/dst  = [M, N, 1, 1]
    // C memory index: m + n*M
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[k + m*K] * B[k + n*K];
            }
            C[m + n*M] = acc;
        }
    }
}

int matmul_tsi_test(int argc, char **argv) {
    int K = 256;
    int M = 8;
    int N = 64;

    matmul_tsi_parse_shape_args(argc, argv, &K, &M, &N);

    fprintf(stderr,
            "\nMAT_MUL test: A=[%d,%d,1,1] B=[%d,%d,1,1] C=[%d,%d,1,1]\n",
            K, M, K, N, M, N);

    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "ERROR: invalid shape K=%d M=%d N=%d\n", K, M, N);
        return -1;
    }

    std::vector<float> A((size_t)K * (size_t)M);
    std::vector<float> B((size_t)K * (size_t)N);
    std::vector<float> C((size_t)M * (size_t)N, 0.0f);
    std::vector<float> C_ref((size_t)M * (size_t)N, 0.0f);

    matmul_tsi_fill_inputs(A, B, K, M, N);
    matmul_tsi_reference(A, B, C_ref, K, M, N);

    ggml_time_init();

    ggml_backend_t backend = ggml_backend_tsavorite_init();
    if (!backend) {
        fprintf(stderr, "ERROR: ggml_backend_tsavorite_init failed\n");
        return -1;
    }

    // This context owns GGML metadata for A/B/C graph nodes.
    // Tensor payloads are allocated by backend/gallocr because no_alloc=true.
    const size_t ctx_size =
        ggml_tensor_overhead() * 16 +
        ggml_graph_overhead() +
        1024 * 1024;

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ERROR: ggml_init failed\n");
        ggml_backend_free(backend);
        return -1;
    }

    // Create GGML MAT_MUL inputs exactly as backend expects:
    //   A=[K,M,1,1], B=[K,N,1,1]
    struct ggml_tensor *a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, M, 1, 1);
    struct ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, N, 1, 1);

    if (!a || !b) {
        fprintf(stderr, "ERROR: failed to create MAT_MUL tensors\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!input_buffer) {
        fprintf(stderr, "ERROR: ggml_backend_alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    ggml_backend_tensor_set(a, A.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, B.data(), 0, ggml_nbytes(b));

    // This is the GGML MAT_MUL API call under test.
    // Expected output shape is [M,N,1,1].
    struct ggml_tensor *c = ggml_mul_mat(ctx, a, b);
    if (!c) {
        fprintf(stderr, "ERROR: ggml_mul_mat failed\n");
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    if (c->ne[0] != M || c->ne[1] != N || c->ne[2] != 1 || c->ne[3] != 1) {
        fprintf(stderr,
                "ERROR: unexpected MAT_MUL output shape C=[%ld,%ld,%ld,%ld], expected [%d,%d,1,1]\n",
                (long)c->ne[0], (long)c->ne[1], (long)c->ne[2], (long)c->ne[3], M, N);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    if (!gf) {
        fprintf(stderr, "ERROR: ggml_new_graph failed\n");
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    ggml_build_forward_expand(gf, c);

    ggml_gallocr_t allocr =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    if (!allocr) {
        fprintf(stderr, "ERROR: ggml_gallocr_new failed\n");
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    ggml_gallocr_reserve(allocr, gf);
    fprintf(stderr, "MAT_MUL compute buffer size: %.4f KB\n",
            ggml_gallocr_get_buffer_size(allocr, 0) / 1024.0);

    ggml_gallocr_alloc_graph(allocr, gf);

    const enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: ggml_backend_graph_compute failed status=%d\n", status);
        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return -1;
    }

    ggml_backend_tensor_get(c, C.data(), 0, ggml_nbytes(c));

    int mismatches = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!matmul_tsi_float_equal(C[i], C_ref[i])) {
            if (mismatches < 32) {
                fprintf(stderr,
                        "Mismatch index=%d got=%f expected=%f\n",
                        i, C[i], C_ref[i]);
            }
            ++mismatches;
        }
    }

    if (mismatches) {
        fprintf(stderr, "\nMAT_MUL TEST FAILED mismatches=%d total=%d\n",
                mismatches, M * N);
    } else {
        fprintf(stderr, "\nMAT_MUL TEST PASSED\n");
    }

    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return mismatches ? -1 : 0;
}

#endif // MAT_MUL_TSI_TEST_CASE_CPP

