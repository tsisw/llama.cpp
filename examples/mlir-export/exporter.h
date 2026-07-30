// Shared ggml-graph-to-linalg-MLIR exporter machinery, used by the separate test-case
// programs in this directory (mlir-export-matmul.cpp, mlir-export-matmul-add.cpp).
//
// v1 scope: GGML_OP_MUL_MAT and GGML_OP_ADD only, GGML_TYPE_F32 only, single-output
// graphs only, no broadcasting (ADD requires exact-matching shapes).
// See docs/superpowers/specs/2026-07-06-ggml-mlir-export-design.md for the design.
//
// Emits MLIR text matching the entry-point conventions expected by the TSI compiler
// (tsi-opt / txe dialect): func name, `txe.name` argument/result attributes,
// `llvm.emit_c_interface`.

#pragma once

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------------------
// linalg exporter
// ----------------------------------------------------------------------------------------

// MLIR shape is (ne[n_dims-1], ..., ne[0]) for every tensor, including computed nodes -
// this is what makes the mapping consistent with ggml's own ne[0]=cols/ne[1]=rows
// convention, and it's why MUL_MAT's output ends up "transposed" relative to the
// intuitive A*B^T result (see the design doc).
// Thrown (instead of exit(1)) when the graph uses something the exporter can't emit yet, so a
// caller embedding the exporter (e.g. the ggml-tsavorite backend's whole-graph capture) can catch
// it and fall back gracefully instead of the process aborting mid-run. The specific reason has
// already been printed to stderr at the throw site.
struct mlir_export_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::string mlir_element_type(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return "f32";
    }
    if (t->type == GGML_TYPE_BF16) {
        return "bf16";
    }
    if (t->type == GGML_TYPE_I32) {
        return "i32";
    }
    fprintf(stderr, "mlir-export: unsupported tensor type: %s\n", ggml_type_name(t->type));
    throw mlir_export_error("unsupported graph construct (see message above)");
}

static std::string mlir_shape_dims(const ggml_tensor * t) {
    int n_dims = ggml_n_dims(t);
    std::ostringstream oss;
    for (int i = n_dims - 1; i >= 0; i--) {
        oss << t->ne[i] << "x";
    }
    return oss.str();
}

static std::string mlir_tensor_type(const ggml_tensor * t) {
    return "tensor<" + mlir_shape_dims(t) + mlir_element_type(t) + ">";
}

// MLIR type using an explicit rank R (>= ggml_n_dims): ggml's trailing size-1 dims become MLIR
// leading size-1 dims. Used by CONCAT, where an operand can report a lower ggml_n_dims than the
// node (e.g. the decode graph's new K/V [.,.,1] vs the cache [.,.,cur]).
static std::string mlir_tensor_type_ranked(const ggml_tensor * t, int R) {
    std::ostringstream oss;
    oss << "tensor<";
    for (int i = R - 1; i >= 0; i--) oss << t->ne[i] << "x";
    oss << mlir_element_type(t) << ">";
    return oss.str();
}

// tensor type for the explicit transpose of a 2D tensor (swaps ne[0] and ne[1])
static std::string mlir_transposed_tensor_type(const ggml_tensor * t) {
    return "tensor<" + std::to_string(t->ne[0]) + "x" + std::to_string(t->ne[1]) + "x" + mlir_element_type(t) + ">";
}

// formats a float as an MLIR floating-point literal, e.g. "60" -> "60.0", "1e-06" unchanged.
// +-inf use the same hex bit-pattern form already proven to parse/lower correctly elsewhere
// in this file (emit_soft_max's row-max fill), rather than a textual "-inf" token whose
// parsing inside a dense<> literal isn't something this codebase has exercised before.
static std::string format_f32_literal(float v) {
    if (std::isinf(v)) {
        return v < 0 ? "0xFF800000" : "0x7F800000";
    }
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", (double) v);
    std::string s(buf);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
        s.find("inf") == std::string::npos && s.find("nan") == std::string::npos) {
        s += ".0";
    }
    return s;
}

// ----------------------------------------------------------------------------------------
// N-D affine-map / iterator-type helpers, shared by every linalg.generic emitter below.
// `keep` selects, in order, which of the (d0..d_{n_dims-1}) iteration-space dims appear in
// an operand's own index space - e.g. affine_map_select(3, {0,2}) drops the middle dim,
// which is exactly the "broadcast over one batch dim" shape ROPE's rotate step needs.
// ----------------------------------------------------------------------------------------

static std::string affine_map_select(int n_dims, const std::vector<int> & keep) {
    std::ostringstream oss;
    oss << "affine_map<(";
    for (int i = 0; i < n_dims; i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << "d" << i;
    }
    oss << ") -> (";
    for (size_t i = 0; i < keep.size(); i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << "d" << keep[i];
    }
    oss << ")>";
    return oss.str();
}

static std::vector<int> dims_range(int n_dims) {
    std::vector<int> v(n_dims);
    for (int i = 0; i < n_dims; i++) {
        v[i] = i;
    }
    return v;
}

// (d0..dn-1) -> (d0..dn-1): identity map over the full iteration space.
static std::string affine_map_full(int n_dims) {
    return affine_map_select(n_dims, dims_range(n_dims));
}

// (d0..dn-1) -> (d0..dn-2): drops the innermost dim, e.g. for a per-row reduction result
// broadcast back across the full shape.
static std::string affine_map_drop_last(int n_dims) {
    std::vector<int> keep = dims_range(n_dims);
    keep.pop_back();
    return affine_map_select(n_dims, keep);
}

static std::string iterator_types_all(int n_dims, const char * kind) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < n_dims; i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << "\"" << kind << "\"";
    }
    oss << "]";
    return oss.str();
}

static std::string iterator_types_all_parallel(int n_dims) {
    return iterator_types_all(n_dims, "parallel");
}

// [parallel]*n_dims-1 + [reduction]: reduces only the innermost dim.
static std::string iterator_types_reduce_last(int n_dims) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < n_dims; i++) {
        if (i > 0) {
            oss << ", ";
        }
        oss << (i == n_dims - 1 ? "\"reduction\"" : "\"parallel\"");
    }
    oss << "]";
    return oss.str();
}

// tensor type for the reduction of the innermost (ne[0]/cols) dim of an N-D tensor, e.g. a
// (tokens, heads, hidden) tensor reduced over hidden becomes (tokens, heads).
static std::string mlir_reduced_tensor_type(const ggml_tensor * t) {
    int n_dims = ggml_n_dims(t);
    std::ostringstream oss;
    oss << "tensor<";
    for (int i = n_dims - 1; i >= 1; i--) {
        oss << t->ne[i] << "x";
    }
    oss << mlir_element_type(t) << ">";
    return oss.str();
}

struct linalg_exporter {
    std::ostringstream                        body;
    std::map<const ggml_tensor *, std::string> values;
    int                                        next_id = 0;
    bool                                       cst_emitted = false;

    std::string new_id() {
        return "%" + std::to_string(next_id++);
    }

    void ensure_cst() {
        if (!cst_emitted) {
            body << "    %cst = arith.constant 0.000000e+00 : f32\n";
            cst_emitted = true;
        }
    }

    // ggml_mul_mat(A, B) = A * B^T, and ggml stores the raw result with
    // ne[0] = A.ne[1], ne[1] = B.ne[1] - i.e. (read with the usual ne[0]=cols/ne[1]=rows
    // convention) it is the transpose of the intuitive result. To keep every tensor in the
    // exported graph following that same convention, we transpose A (not B) and emit
    // linalg.matmul(B, transpose(A)) -> shape (B.ne[1], A.ne[1]).
    //
    // Dispatches on rank: 2D uses the original (proven) linalg.matmul path unchanged. 3D
    // batched matmul comes in two forms, matching ggml's own ggml_can_mul_mat/GQA rule
    // (t1->ne[2] % t0->ne[2] == 0, see ggml-cpu.c's mul_mat compute kernel: r2 = ne12/ne02,
    // i02 = i12/r2 - i.e. b's heads are grouped in contiguous blocks of size r2, each block
    // sharing one a-head): equal head counts need no broadcast, and unequal-but-divisible
    // head counts (real attention's Q/KV head mismatch, e.g. TinyLlama's 32 Q : 4 KV heads)
    // repeat a's heads to match b's before the same batched matmul.
    std::string emit_mul_mat(const ggml_tensor * node) {
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];

        int a_dims = ggml_n_dims(a);
        int b_dims = ggml_n_dims(b);

        if (a_dims == 2 && b_dims == 2) {
            return emit_mul_mat_2d(node, a, b);
        }
        if (a_dims == 2 && b_dims == 1) {   // n_tokens=1 decode: b is ggml [k,1] collapsed to rank-1
            return emit_mul_mat_2d_vec(node, a, b);
        }
        if (a_dims == 3 && b_dims == 3) {
            if (a->ne[2] == b->ne[2]) {
                return emit_mul_mat_batched_3d(node, a, b);
            }
            if (b->ne[2] % a->ne[2] == 0) {
                return emit_mul_mat_batched_3d_gqa(node, a, b);
            }
        }
        fprintf(stderr,
                "mlir-export: MUL_MAT only supports 2D, 3D with equal batch dims, or 3D GQA broadcast "
                "(b's head count a multiple of a's)\n");
        throw mlir_export_error("unsupported graph construct (see message above)");
    }

    // ggml_mul_mat(a[k,n], b[k,1]) with a single column (n_tokens=1): b is rank-1 [k], result rank-1
    // [n]. Expand b to [1,k], reuse the transpose+matmul path, then collapse [1,n] -> [n].
    std::string emit_mul_mat_2d_vec(const ggml_tensor * node, const ggml_tensor * a, const ggml_tensor * b) {
        ensure_cst();
        const int64_t k = b->ne[0];
        const int64_t n = a->ne[1];

        // bf16 weight: down-cast the activation so both matmul inputs are bf16 (see emit_mul_mat_2d).
        std::string bsrc = values.at(b), belem = mlir_element_type(b);
        if (a->type == GGML_TYPE_BF16 && b->type == GGML_TYPE_F32) {
            bsrc  = cast_f32_to_bf16(b, values.at(b));
            belem = "bf16";
        }

        std::string b2 = new_id();
        body << "    " << b2 << " = tensor.expand_shape " << bsrc << " [[0, 1]] output_shape [1, " << k
             << "] : tensor<" << k << "x" << belem << "> into tensor<1x" << k << "x" << belem << ">\n";

        std::string at_ty = mlir_transposed_tensor_type(a);   // [k, n]
        std::string at_init = new_id();
        body << "    " << at_init << " = tensor.empty() : " << at_ty << "\n";
        std::string at = new_id();
        body << "    " << at << " = linalg.transpose ins(" << values.at(a) << " : " << mlir_tensor_type(a)
             << ") outs(" << at_init << " : " << at_ty << ") permutation = [1, 0]\n";

        std::string mm_ty = "tensor<1x" + std::to_string(n) + "x" + mlir_element_type(node) + ">";
        std::string mm_init = new_id();
        body << "    " << mm_init << " = tensor.empty() : " << mm_ty << "\n";
        std::string filled = new_id();
        body << "    " << filled << " = linalg.fill ins(%cst : f32) outs(" << mm_init << " : " << mm_ty << ") -> " << mm_ty << "\n";
        std::string mm = new_id();
        body << "    " << mm << " = linalg.matmul ins(" << b2 << ", " << at << " : tensor<1x" << k << "x" << belem
             << ">, " << at_ty << ") outs(" << filled << " : " << mm_ty << ") -> " << mm_ty << "\n";

        std::string result = new_id();
        body << "    " << result << " = tensor.collapse_shape " << mm << " [[0, 1]] : " << mm_ty << " into "
             << mlir_tensor_type(node) << "\n";
        return result;
    }

    // elementwise f32 -> bf16 (arith.truncf); returns the new SSA value id
    std::string cast_f32_to_bf16(const ggml_tensor * t, const std::string & val) {
        int R = ggml_n_dims(t);
        std::string bf16_ty = "tensor<" + mlir_shape_dims(t) + "bf16>";
        std::string init = new_id();
        body << "    " << init << " = tensor.empty() : " << bf16_ty << "\n";
        std::string out = new_id();
        body << "    " << out << " = linalg.generic {indexing_maps = [" << affine_map_full(R) << ", "
             << affine_map_full(R) << "], iterator_types = " << iterator_types_all_parallel(R) << "} ins("
             << val << " : " << mlir_tensor_type(t) << ") outs(" << init << " : " << bf16_ty << ") {\n";
        body << "    ^bb0(%in: f32, %o: bf16):\n";
        body << "      %c = arith.truncf %in : f32 to bf16\n";
        body << "      linalg.yield %c : bf16\n";
        body << "    } -> " << bf16_ty << "\n";
        return out;
    }

    std::string emit_mul_mat_2d(const ggml_tensor * node, const ggml_tensor * a, const ggml_tensor * b) {
        ensure_cst();

        const std::string & a_val = values.at(a);
        const std::string & b_val = values.at(b);

        std::string at_ty   = mlir_transposed_tensor_type(a);
        std::string at_init = new_id();
        body << "    " << at_init << " = tensor.empty() : " << at_ty << "\n";
        std::string at = new_id();
        body << "    " << at << " = linalg.transpose ins(" << a_val << " : " << mlir_tensor_type(a)
             << ") outs(" << at_init << " : " << at_ty << ") permutation = [1, 0]\n";

        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";

        std::string filled = new_id();
        body << "    " << filled << " = linalg.fill ins(%cst : f32) outs(" << out_init << " : "
             << out_ty << ") -> " << out_ty << "\n";

        // bf16 weight: the TMU needs both matmul inputs the same type, so down-cast the f32
        // activation to bf16 and matmul bf16 x bf16 -> f32 accumulate (bf16 on the TMU).
        std::string lhs = b_val, lhs_ty = mlir_tensor_type(b);
        if (a->type == GGML_TYPE_BF16 && b->type == GGML_TYPE_F32) {
            lhs = cast_f32_to_bf16(b, b_val);
            lhs_ty = "tensor<" + mlir_shape_dims(b) + "bf16>";
        }

        std::string result = new_id();
        body << "    " << result << " = linalg.matmul ins(" << lhs << ", " << at << " : "
             << lhs_ty << ", " << at_ty << ") outs(" << filled << " : "
             << out_ty << ") -> " << out_ty << "\n";

        return result;
    }

    // Batched matmul (Phase 4, step 1): a, b are both (batch=H, rows, cols) in MLIR shape
    // order, H equal on both operands (no GQA broadcast). Same transpose-A-then-matmul
    // structure as emit_mul_mat_2d, generalized by adding a leading batch dim that both the
    // transpose permutation and linalg.batch_matmul carry straight through unchanged per
    // batch entry - i.e. for each h in [0,H), out[h] = mul_mat_2d(a[h], b[h]).
    std::string emit_mul_mat_batched_3d(const ggml_tensor * node, const ggml_tensor * a, const ggml_tensor * b) {
        return emit_mul_mat_batched_3d_core(node, values.at(a), a->ne[0], a->ne[1], a->ne[2], b);
    }

    // GQA broadcast (Phase 4, step 2): b has more heads than a, in a multiple (b->ne[2] % a
    // ->ne[2] == 0). Repeats a's heads to match b's head count first (via
    // emit_repeat_heads_3d), then runs the exact same batched-matmul core as the equal-heads
    // case above - no new matmul logic, just a pre-broadcast of the smaller operand.
    std::string emit_mul_mat_batched_3d_gqa(const ggml_tensor * node, const ggml_tensor * a, const ggml_tensor * b) {
        int64_t     H           = b->ne[2];
        std::string a_repeated  = emit_repeat_heads_3d(a, H);
        return emit_mul_mat_batched_3d_core(node, a_repeated, a->ne[0], a->ne[1], H, b);
    }

    // Repeats a rank-3 (H_src, rows, cols) tensor's heads into an (H_dst, rows, cols) tensor,
    // H_dst a multiple of H_src, matching ggml-cpu's own GQA grouping (contiguous blocks of
    // size H_dst/H_src all sharing one source head - see emit_mul_mat's comment). Built
    // entirely from tensor.extract_slice/insert_slice (already proven through the real TSI
    // pipeline by ROPE's deinterleave/reinterleave), rather than a linalg.generic with a
    // floordiv indexing map, which is valid MLIR but untested through this pipeline's
    // tile/vectorize stages - given how many "valid-but-not-accepted-by-this-pipeline"
    // surprises RoPE's broadcast turned up, this sticks to already-proven primitives.
    std::string emit_repeat_heads_3d(const ggml_tensor * a, int64_t target_h) {
        int64_t     src_h = a->ne[2];
        int64_t     group = target_h / src_h;
        int64_t     ne1   = a->ne[1];
        int64_t     ne0   = a->ne[0];
        std::string elem  = mlir_element_type(a);
        std::string a_ty  = mlir_tensor_type(a);
        std::string out_ty =
            "tensor<" + std::to_string(target_h) + "x" + std::to_string(ne1) + "x" + std::to_string(ne0) + "x" +
            elem + ">";
        std::string slice_ty = "tensor<1x" + std::to_string(ne1) + "x" + std::to_string(ne0) + "x" + elem + ">";

        const std::string & a_val = values.at(a);

        std::string current = new_id();
        body << "    " << current << " = tensor.empty() : " << out_ty << "\n";

        for (int64_t h_src = 0; h_src < src_h; h_src++) {
            std::string slice = new_id();
            body << "    " << slice << " = tensor.extract_slice " << a_val << "[" << h_src << ", 0, 0] [1, " << ne1
                 << ", " << ne0 << "] [1, 1, 1] : " << a_ty << " to " << slice_ty << "\n";
            for (int64_t g = 0; g < group; g++) {
                int64_t     h_dst = h_src * group + g;
                std::string next  = new_id();
                body << "    " << next << " = tensor.insert_slice " << slice << " into " << current << "[" << h_dst
                     << ", 0, 0] [1, " << ne1 << ", " << ne0 << "] [1, 1, 1] : " << slice_ty << " into " << out_ty
                     << "\n";
                current = next;
            }
        }

        return current;
    }

    std::string emit_mul_mat_batched_3d_core(const ggml_tensor * node, const std::string & a_val, int64_t a_ne0,
                                              int64_t a_ne1, int64_t H, const ggml_tensor * b) {
        ensure_cst();

        const std::string & b_val = values.at(b);

        std::string elem = mlir_element_type(b);
        std::string a_ty =
            "tensor<" + std::to_string(H) + "x" + std::to_string(a_ne1) + "x" + std::to_string(a_ne0) + "x" + elem +
            ">";
        std::string b_ty = mlir_tensor_type(b);
        std::string at_ty =
            "tensor<" + std::to_string(H) + "x" + std::to_string(a_ne0) + "x" + std::to_string(a_ne1) + "x" + elem +
            ">";

        std::string at_init = new_id();
        body << "    " << at_init << " = tensor.empty() : " << at_ty << "\n";
        std::string at = new_id();
        body << "    " << at << " = linalg.transpose ins(" << a_val << " : " << a_ty << ") outs(" << at_init
             << " : " << at_ty << ") permutation = [0, 2, 1]\n";

        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";

        std::string filled = new_id();
        body << "    " << filled << " = linalg.fill ins(%cst : f32) outs(" << out_init << " : " << out_ty
             << ") -> " << out_ty << "\n";

        std::string result = new_id();
        body << "    " << result << " = linalg.batch_matmul ins(" << b_val << ", " << at << " : " << b_ty << ", "
             << at_ty << ") outs(" << filled << " : " << out_ty << ") -> " << out_ty << "\n";

        return result;
    }

    // shared by emit_add/emit_mul. Two shapes supported: identical shape (any rank -
    // linalg.add/linalg.mul are named ops that infer identity indexing maps from the operand
    // shapes, no explicit affine maps needed), or a's innermost-dim broadcast against a 1D b
    // matching that dim - the real rms_norm(x)*weight pattern (a per-channel norm weight
    // applied identically to every token/head/batch entry), matching ggml's own
    // ggml_can_repeat(b,a) rule for the case where b's non-innermost dims are all 1. Other
    // broadcast shapes (e.g. b repeating a non-1 number of times) are still unsupported.
    std::string emit_elementwise_binop(const ggml_tensor * node, const char * linalg_op, const char * ggml_op_name) {
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];

        int n_dims = ggml_n_dims(a);
        bool same_shape = ggml_n_dims(b) == n_dims;
        for (int i = 0; same_shape && i < n_dims; i++) {
            same_shape = a->ne[i] == b->ne[i];
        }

        const std::string & a_val = values.at(a);
        const std::string & b_val = values.at(b);
        std::string          ty   = mlir_tensor_type(node);

        if (same_shape) {
            std::string out_init = new_id();
            body << "    " << out_init << " = tensor.empty() : " << ty << "\n";
            std::string result = new_id();
            body << "    " << result << " = linalg." << linalg_op << " ins(" << a_val << ", " << b_val << " : " << ty
                 << ", " << ty << ") outs(" << out_init << " : " << ty << ") -> " << ty << "\n";
            return result;
        }

        if (n_dims >= 2 && ggml_n_dims(b) == 1 && b->ne[0] == a->ne[0]) {
            std::string b_ty     = mlir_tensor_type(b);
            std::string full_map = affine_map_full(n_dims);
            std::string last_map = affine_map_select(n_dims, { n_dims - 1 });
            std::string full_it  = iterator_types_all_parallel(n_dims);
            const char * arith_op = std::string(linalg_op) == "add" ? "addf" : "mulf";

            std::string out_init = new_id();
            body << "    " << out_init << " = tensor.empty() : " << ty << "\n";
            std::string result = new_id();
            body << "    " << result << " = linalg.generic {indexing_maps = [" << full_map << ", " << last_map
                 << ", " << full_map << "], iterator_types = " << full_it << "} ins(" << a_val << ", " << b_val
                 << " : " << ty << ", " << b_ty << ") outs(" << out_init << " : " << ty << ") {\n";
            body << "    ^bb0(%in: f32, %w: f32, %out: f32):\n";
            body << "      %r = arith." << arith_op << " %in, %w : f32\n";
            body << "      linalg.yield %r : f32\n";
            body << "    } -> " << ty << "\n";
            return result;
        }

        fprintf(stderr,
                "mlir-export: %s only supports equal-shape tensors, or a's innermost-dim broadcast against a "
                "matching 1D tensor, for now\n",
                ggml_op_name);
        throw mlir_export_error("unsupported graph construct (see message above)");
    }

    std::string emit_add(const ggml_tensor * node) {
        return emit_elementwise_binop(node, "add", "ADD");
    }

    std::string emit_mul(const ggml_tensor * node) {
        return emit_elementwise_binop(node, "mul", "MUL");
    }

    // out = in * s, where s is a compile-time constant from node->op_params[0]. Only
    // supports plain ggml_scale (bias must be 0 - ggml_scale_bias's non-zero-bias variant,
    // at op_params[1], is unsupported for now).
    std::string emit_scale(const ggml_tensor * node) {
        const ggml_tensor * x = node->src[0];

        float s, b;
        memcpy(&s, node->op_params, sizeof(float));
        memcpy(&b, (const char *) node->op_params + sizeof(float), sizeof(float));
        if (b != 0.0f) {
            fprintf(stderr, "mlir-export: SCALE only supports zero bias for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        const std::string & x_val = values.at(x);
        std::string          ty   = mlir_tensor_type(node);
        int                  n_dims = ggml_n_dims(node);
        std::string          id_map = affine_map_full(n_dims);
        std::string          iters  = iterator_types_all_parallel(n_dims);

        std::string s_val = new_id();
        body << "    " << s_val << " = arith.constant " << format_f32_literal(s) << " : f32\n";

        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << ty << "\n";

        std::string result = new_id();
        body << "    " << result << " = linalg.generic {indexing_maps = [" << id_map << ", " << id_map
             << "], iterator_types = " << iters << "} ins(" << x_val << " : " << ty << ") outs(" << out_init << " : "
             << ty << ") {\n";
        body << "    ^bb0(%in: f32, %out: f32):\n";
        body << "      %r = arith.mulf %in, " << s_val << " : f32\n";
        body << "      linalg.yield %r : f32\n";
        body << "    } -> " << ty << "\n";

        return result;
    }

    // SiLU(x) = x / (1 + exp(-x)) - matches ggml's ggml_silu_f32 exactly (see
    // src/ggml-cpu/vec.h).
    std::string emit_silu(const ggml_tensor * node) {
        const ggml_tensor * x = node->src[0];

        const std::string & x_val = values.at(x);
        std::string          ty   = mlir_tensor_type(node);
        int                  n_dims = ggml_n_dims(node);
        std::string          id_map = affine_map_full(n_dims);
        std::string          iters  = iterator_types_all_parallel(n_dims);

        std::string one_val = new_id();
        body << "    " << one_val << " = arith.constant 1.000000e+00 : f32\n";

        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << ty << "\n";

        std::string result = new_id();
        body << "    " << result << " = linalg.generic {indexing_maps = [" << id_map << ", " << id_map
             << "], iterator_types = " << iters << "} ins(" << x_val << " : " << ty << ") outs(" << out_init << " : "
             << ty << ") {\n";
        body << "    ^bb0(%in: f32, %out: f32):\n";
        body << "      %neg = arith.negf %in : f32\n";
        body << "      %e = math.exp %neg : f32\n";
        body << "      %denom = arith.addf " << one_val << ", %e : f32\n";
        body << "      %r = arith.divf %in, %denom : f32\n";
        body << "      linalg.yield %r : f32\n";
        body << "    } -> " << ty << "\n";

        return result;
    }

    // ggml_rms_norm(x, eps) = x / sqrt(mean(x^2, axis=ne0) + eps), applied independently per
    // row (each row of length ne[0]/cols is its own normalization group - matches ggml's own
    // CPU reference, which reduces over ne00, the innermost/last-MLIR dim). This is the first
    // *reduction* op the exporter emits, done as three linalg.generic ops: sum-of-squares per
    // row, then a pointwise mean+eps+rsqrt to get a per-row scale, then a broadcast-multiply.
    // eps comes from node->op_params[0] (ggml stores it as a raw float there).
    std::string emit_rms_norm(const ggml_tensor * node) {
        const ggml_tensor * x = node->src[0];

        int n_dims = ggml_n_dims(x);   // rank 1 (n_tokens=1 decode) reduces to a rank-0 scalar; valid.

        float eps;
        memcpy(&eps, node->op_params, sizeof(float));

        ensure_cst();

        const std::string & x_val = values.at(x);
        std::string          x_ty = mlir_tensor_type(x);
        int64_t              cols = x->ne[0];
        std::string          vec_ty     = mlir_reduced_tensor_type(x);
        std::string          full_map   = affine_map_full(n_dims);
        std::string          drop_map   = affine_map_drop_last(n_dims);
        std::string          reduce_it  = iterator_types_reduce_last(n_dims);
        std::string          full_it    = iterator_types_all_parallel(n_dims);
        std::string          vec_id_map = affine_map_full(n_dims - 1);
        std::string          vec_it     = iterator_types_all_parallel(n_dims - 1);

        std::string ninv_val = new_id();
        body << "    " << ninv_val << " = arith.constant " << format_f32_literal(1.0f / (float) cols) << " : f32\n";
        std::string eps_val = new_id();
        body << "    " << eps_val << " = arith.constant " << format_f32_literal(eps) << " : f32\n";

        // 1. sum of squares per row (reduction over the innermost/column dim, all other dims
        // treated as independent batch dims)
        std::string sum_init = new_id();
        body << "    " << sum_init << " = tensor.empty() : " << vec_ty << "\n";
        std::string sum_filled = new_id();
        body << "    " << sum_filled << " = linalg.fill ins(%cst : f32) outs(" << sum_init << " : " << vec_ty
             << ") -> " << vec_ty << "\n";

        std::string sumsq = new_id();
        body << "    " << sumsq << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map
             << "], iterator_types = " << reduce_it << "} ins(" << x_val << " : " << x_ty << ") outs(" << sum_filled
             << " : " << vec_ty << ") {\n";
        body << "    ^bb0(%in: f32, %acc: f32):\n";
        body << "      %sq = arith.mulf %in, %in : f32\n";
        body << "      %newacc = arith.addf %acc, %sq : f32\n";
        body << "      linalg.yield %newacc : f32\n";
        body << "    } -> " << vec_ty << "\n";

        // 2. per-row scale = rsqrt(mean + eps)
        std::string scale_init = new_id();
        body << "    " << scale_init << " = tensor.empty() : " << vec_ty << "\n";
        std::string scale = new_id();
        body << "    " << scale << " = linalg.generic {indexing_maps = [" << vec_id_map << ", " << vec_id_map
             << "], iterator_types = " << vec_it << "} ins(" << sumsq << " : " << vec_ty << ") outs(" << scale_init
             << " : " << vec_ty << ") {\n";
        body << "    ^bb0(%in: f32, %out: f32):\n";
        body << "      %mean = arith.mulf %in, " << ninv_val << " : f32\n";
        body << "      %meaneps = arith.addf %mean, " << eps_val << " : f32\n";
        body << "      %rs = math.rsqrt %meaneps : f32\n";
        body << "      linalg.yield %rs : f32\n";
        body << "    } -> " << vec_ty << "\n";

        // 3. broadcast-multiply: out[...,c] = x[...,c] * scale[...]
        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";
        std::string result = new_id();
        body << "    " << result << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map << ", "
             << full_map << "], iterator_types = " << full_it << "} ins(" << x_val << ", " << scale << " : " << x_ty
             << ", " << vec_ty << ") outs(" << out_init << " : " << out_ty << ") {\n";
        body << "    ^bb0(%in: f32, %sc: f32, %out: f32):\n";
        body << "      %m = arith.mulf %in, %sc : f32\n";
        body << "      linalg.yield %m : f32\n";
        body << "    } -> " << out_ty << "\n";

        return result;
    }

    // ggml_soft_max_ext(x, mask, scale, max_bias) = softmax(x*scale + mask), reduced over
    // ne0/cols (see ggml-cpu.c's compute kernel: wp = x*scale; wp += slope*mask; slope=1 when
    // max_bias=0). Supports: no mask (scale still applied), or a mask when x is rank 3
    // (n_head, q_rows, kv_cols) and mask is rank 2 (q_rows, kv_cols), broadcasting the mask
    // identically across every head (matching ggml's i12 = i02 % mask_ne2 with mask_ne2=1) -
    // exactly the shape real causal-mask attention needs. ALiBi (max_bias != 0) is out of
    // scope. Decomposed as: an optional stage 0 (scale [+ mask]) feeding the same four-stage
    // reduction (row-max, broadcast-subtract+exp, row-sum, broadcast-divide) used before -
    // when scale=1 and there's no mask, stage 0 is skipped entirely and the original
    // (already-proven) codegen is emitted byte-for-byte, so no regression risk for existing
    // callers.
    std::string emit_soft_max(const ggml_tensor * node) {
        const ggml_tensor * x    = node->src[0];
        const ggml_tensor * mask = node->src[1];

        float scale, max_bias;
        memcpy(&scale, node->op_params, sizeof(float));
        memcpy(&max_bias, (const char *) node->op_params + sizeof(float), sizeof(float));
        if (max_bias != 0.0f) {
            fprintf(stderr, "mlir-export: SOFT_MAX only supports max_bias=0 (no ALiBi) for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        int n_dims = ggml_n_dims(x);
        if (n_dims < 2) {
            fprintf(stderr, "mlir-export: SOFT_MAX requires at least 2 dims\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }
        // mask [n_kv, n_q] broadcast across the head dim (d0). For n_q=1 (decode) it collapses to a
        // rank-1 [n_kv] mask; accept that too, broadcasting across both the head and query dims.
        bool mask_ok = false;
        if (mask != nullptr && n_dims == 3 && mask->ne[0] == x->ne[0]) {
            if (ggml_n_dims(mask) == 2 && mask->ne[1] == x->ne[1]) {
                mask_ok = true;
            } else if (ggml_n_dims(mask) == 1 && x->ne[1] == 1) {
                mask_ok = true;
            }
        }
        if (mask != nullptr && !mask_ok) {
            fprintf(stderr,
                    "mlir-export: SOFT_MAX with a mask only supports rank-3 x with a matching rank-2 mask "
                    "(or a rank-1 [n_kv] mask when n_q=1) broadcast across the head dim for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        ensure_cst();

        const std::string & x_val = values.at(x);
        std::string          x_ty = mlir_tensor_type(x);
        std::string          vec_ty    = mlir_reduced_tensor_type(x);
        std::string          full_map  = affine_map_full(n_dims);
        std::string          drop_map  = affine_map_drop_last(n_dims);
        std::string          reduce_it = iterator_types_reduce_last(n_dims);
        std::string          full_it   = iterator_types_all_parallel(n_dims);

        // stage 0: combined = x*scale [+ mask, broadcast over the head dim]. Skipped (using
        // x_val directly) when it would be a no-op, to keep the already-proven scale=1/
        // no-mask codegen path byte-for-byte unchanged.
        std::string combined_val;
        if (mask == nullptr && scale == 1.0f) {
            combined_val = x_val;
        } else {
            std::string scale_val = new_id();
            body << "    " << scale_val << " = arith.constant " << format_f32_literal(scale) << " : f32\n";
            std::string comb_init = new_id();
            body << "    " << comb_init << " = tensor.empty() : " << x_ty << "\n";
            combined_val = new_id();
            if (mask != nullptr) {
                const std::string & mask_val = values.at(mask);
                std::string          mask_ty  = mlir_tensor_type(mask);
                // rank-2 mask [n_q,n_kv] -> x dims {1,2}; rank-1 mask [n_kv] (n_q=1) -> x dim {2}
                std::string          mask_map = (ggml_n_dims(mask) == 1) ? affine_map_select(3, { 2 })
                                                                         : affine_map_select(3, { 1, 2 });
                body << "    " << combined_val << " = linalg.generic {indexing_maps = [" << full_map << ", "
                     << mask_map << ", " << full_map << "], iterator_types = " << full_it << "} ins(" << x_val
                     << ", " << mask_val << " : " << x_ty << ", " << mask_ty << ") outs(" << comb_init << " : "
                     << x_ty << ") {\n";
                body << "    ^bb0(%in: f32, %m: f32, %out: f32):\n";
                body << "      %scaled = arith.mulf %in, " << scale_val << " : f32\n";
                body << "      %c = arith.addf %scaled, %m : f32\n";
                body << "      linalg.yield %c : f32\n";
                body << "    } -> " << x_ty << "\n";
            } else {
                body << "    " << combined_val << " = linalg.generic {indexing_maps = [" << full_map << ", "
                     << full_map << "], iterator_types = " << full_it << "} ins(" << x_val << " : " << x_ty
                     << ") outs(" << comb_init << " : " << x_ty << ") {\n";
                body << "    ^bb0(%in: f32, %out: f32):\n";
                body << "      %scaled = arith.mulf %in, " << scale_val << " : f32\n";
                body << "      linalg.yield %scaled : f32\n";
                body << "    } -> " << x_ty << "\n";
            }
        }

        // 1. row-max reduction (over the innermost dim, all other dims are batch dims)
        std::string neginf_val = new_id();
        body << "    " << neginf_val << " = arith.constant 0xFF800000 : f32\n";
        std::string max_init = new_id();
        body << "    " << max_init << " = tensor.empty() : " << vec_ty << "\n";
        std::string max_filled = new_id();
        body << "    " << max_filled << " = linalg.fill ins(" << neginf_val << " : f32) outs(" << max_init << " : "
             << vec_ty << ") -> " << vec_ty << "\n";
        std::string rowmax = new_id();
        body << "    " << rowmax << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map
             << "], iterator_types = " << reduce_it << "} ins(" << combined_val << " : " << x_ty << ") outs("
             << max_filled << " : " << vec_ty << ") {\n";
        body << "    ^bb0(%in: f32, %acc: f32):\n";
        body << "      %m = arith.maximumf %in, %acc : f32\n";
        body << "      linalg.yield %m : f32\n";
        body << "    } -> " << vec_ty << "\n";

        // 2. exp(x - rowmax), broadcasting rowmax across the innermost dim
        std::string exp_init = new_id();
        body << "    " << exp_init << " = tensor.empty() : " << x_ty << "\n";
        std::string expx = new_id();
        body << "    " << expx << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map << ", "
             << full_map << "], iterator_types = " << full_it << "} ins(" << combined_val << ", " << rowmax << " : "
             << x_ty << ", " << vec_ty << ") outs(" << exp_init << " : " << x_ty << ") {\n";
        body << "    ^bb0(%in: f32, %mx: f32, %out: f32):\n";
        body << "      %sub = arith.subf %in, %mx : f32\n";
        body << "      %e = math.exp %sub : f32\n";
        body << "      linalg.yield %e : f32\n";
        body << "    } -> " << x_ty << "\n";

        // 3. row-sum reduction
        std::string sum_init = new_id();
        body << "    " << sum_init << " = tensor.empty() : " << vec_ty << "\n";
        std::string sum_filled = new_id();
        body << "    " << sum_filled << " = linalg.fill ins(%cst : f32) outs(" << sum_init << " : " << vec_ty
             << ") -> " << vec_ty << "\n";
        std::string rowsum = new_id();
        body << "    " << rowsum << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map
             << "], iterator_types = " << reduce_it << "} ins(" << expx << " : " << x_ty << ") outs(" << sum_filled
             << " : " << vec_ty << ") {\n";
        body << "    ^bb0(%in: f32, %acc: f32):\n";
        body << "      %a = arith.addf %in, %acc : f32\n";
        body << "      linalg.yield %a : f32\n";
        body << "    } -> " << vec_ty << "\n";

        // 4. divide by row-sum, broadcasting across the innermost dim
        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";
        std::string result = new_id();
        body << "    " << result << " = linalg.generic {indexing_maps = [" << full_map << ", " << drop_map << ", "
             << full_map << "], iterator_types = " << full_it << "} ins(" << expx << ", " << rowsum << " : " << x_ty
             << ", " << vec_ty << ") outs(" << out_init << " : " << out_ty << ") {\n";
        body << "    ^bb0(%in: f32, %sm: f32, %out: f32):\n";
        body << "      %d = arith.divf %in, %sm : f32\n";
        body << "      linalg.yield %d : f32\n";
        body << "    } -> " << out_ty << "\n";

        return result;
    }

    // ggml_rope(x, pos, n_dims, GGML_ROPE_TYPE_NORMAL): rotates interleaved pairs (2k,2k+1)
    // of each row by an angle theta_k = pos * freq_base^(-2k/n_dims), matching llama.cpp's
    // LLAMA_ROPE_TYPE_NORM convention used by LLM_ARCH_LLAMA (note this only matches the HF
    // Llama model's own "rotate_half" math because convert_hf_to_gguf.py permutes the Q/K
    // weights specifically so this simpler interleaved-pairs rotation can be used instead -
    // verified against ggml's own ggml_rope() output directly, not just derived by hand).
    //
    // Two supported ranks for x, per ggml's own constraint pos->ne[0] == x->ne[2] (position
    // varies along x's *third* ggml dim; missing dims default to size 1):
    //  - rank 2 (head_dim, n_head): ne[2] is implicitly 1, so pos has exactly one element,
    //    broadcast uniformly across every row (n_head) - a single shared position.
    //  - rank 3 (head_dim, n_head, n_tokens): pos has one element per token (n_tokens =
    //    x->ne[2], the outermost MLIR dim), broadcast across the middle n_head dim - genuine
    //    per-token positions. This is the Phase 2 rank-generalization case.
    // The rank-2 path is kept byte-for-byte as before (already proven through the full real
    // pipeline) to avoid any regression risk; rank 3 is new.
    //
    // Scope restrictions for both ranks: n_dims must equal head_dim (no partial rotation),
    // mode must be GGML_ROPE_TYPE_NORMAL, no YaRN/frequency scaling (freq_scale=1,
    // ext_factor=0, attn_factor=1). Rank 4 (batched sequences) is not yet supported.
    std::string emit_rope(const ggml_tensor * node) {
        const ggml_tensor * x   = node->src[0];
        const ggml_tensor * pos = node->src[1];

        const int32_t * ip = (const int32_t *) node->op_params;
        int32_t         rot_dims = ip[1];
        int32_t         mode     = ip[2];
        float           freq_base, freq_scale, ext_factor, attn_factor;
        memcpy(&freq_base, ip + 5, sizeof(float));
        memcpy(&freq_scale, ip + 6, sizeof(float));
        memcpy(&ext_factor, ip + 7, sizeof(float));
        memcpy(&attn_factor, ip + 8, sizeof(float));

        if (mode != GGML_ROPE_TYPE_NORMAL) {
            fprintf(stderr, "mlir-export: ROPE only supports GGML_ROPE_TYPE_NORMAL for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }
        if (rot_dims != x->ne[0] || rot_dims % 2 != 0) {
            fprintf(stderr, "mlir-export: ROPE only supports full-row rotation (n_dims == ne[0], even) for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }
        if (freq_scale != 1.0f || ext_factor != 0.0f || attn_factor != 1.0f) {
            fprintf(stderr, "mlir-export: ROPE only supports freq_scale=1, ext_factor=0, attn_factor=1 for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        int x_rank = ggml_n_dims(x);
        if (x_rank == 2) {
            return emit_rope_rank2(node, x, pos, rot_dims, freq_base);
        }
        if (x_rank == 3) {
            return emit_rope_rank3(node, x, pos, rot_dims, freq_base);
        }
        fprintf(stderr, "mlir-export: ROPE only supports 2D or 3D x for now (got rank %d)\n", x_rank);
        throw mlir_export_error("unsupported graph construct (see message above)");
    }

    std::string emit_rope_rank2(const ggml_tensor * node, const ggml_tensor * x, const ggml_tensor * pos,
                                 int32_t rot_dims, float freq_base) {
        if (ggml_nelements(pos) != 1) {
            fprintf(stderr, "mlir-export: ROPE (2D x) only supports a single shared position for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        const std::string & x_val   = values.at(x);
        const std::string & pos_val = values.at(pos);
        std::string          x_ty   = mlir_tensor_type(x);
        int64_t               rows  = x->ne[1];
        int64_t               half  = rot_dims / 2;
        std::string          half_ty = "tensor<" + std::to_string(half) + "x" + mlir_element_type(x) + ">";
        std::string          pair_ty = "tensor<" + std::to_string(rows) + "x" + std::to_string(half) + "x" +
                                        mlir_element_type(x) + ">";

        // freq_k = freq_base^(-2k/n_dims), k=0..half-1 - compile-time constant. llama3 rope divides
        // each by freq_factors[k] (src[2]), matching ggml's theta/ff (Llama-3.x long-context scaling).
        const ggml_tensor * ff = node->src[2];
        const float * ffd = (ff && ff->data) ? (const float *) ff->data : nullptr;
        std::ostringstream freq_lit;
        freq_lit << "dense<[";
        for (int64_t k = 0; k < half; k++) {
            if (k > 0) {
                freq_lit << ", ";
            }
            float freq_k = powf(freq_base, -2.0f * (float) k / (float) rot_dims);
            if (ffd) { freq_k /= ffd[k]; }
            freq_lit << format_f32_literal(freq_k);
        }
        freq_lit << "]>";

        std::string freq_val = new_id();
        body << "    " << freq_val << " = arith.constant " << freq_lit.str() << " : " << half_ty << "\n";

        std::string c0 = new_id();
        body << "    " << c0 << " = arith.constant 0 : index\n";
        std::string pos_i32 = new_id();
        body << "    " << pos_i32 << " = tensor.extract " << pos_val << "[" << c0 << "] : " << mlir_tensor_type(pos)
             << "\n";
        std::string pos_f32 = new_id();
        body << "    " << pos_f32 << " = arith.sitofp " << pos_i32 << " : i32 to f32\n";

        // theta_k = pos * freq_k. Uses linalg.fill (to broadcast the scalar pos_f32 into a
        // tensor) + named linalg.mul, rather than capturing pos_f32 as a scalar inside a
        // custom linalg.generic body or using tensor.splat - both alternatives compile and
        // JIT-verify fine generically, but fail in the real TSI pipeline: the scalar-capture
        // form fails to legalize in the final txe-to-LLVM lowering stage (a raw scalar
        // broadcast to a vector register in that shape wasn't accepted), and tensor.splat
        // fails to bufferize at all ("memory space not implemented yet"). linalg.fill and
        // linalg.mul are already proven to lower correctly through the full pipeline (used
        // by emit_mul_mat/emit_rms_norm/emit_soft_max and emit_mul respectively), so
        // building the broadcast from those primitives avoids both problems.
        std::string pos_bcast_init = new_id();
        body << "    " << pos_bcast_init << " = tensor.empty() : " << half_ty << "\n";
        std::string pos_bcast = new_id();
        body << "    " << pos_bcast << " = linalg.fill ins(" << pos_f32 << " : f32) outs(" << pos_bcast_init << " : "
             << half_ty << ") -> " << half_ty << "\n";
        std::string theta_init = new_id();
        body << "    " << theta_init << " = tensor.empty() : " << half_ty << "\n";
        std::string theta = new_id();
        body << "    " << theta << " = linalg.mul ins(" << freq_val << ", " << pos_bcast << " : " << half_ty << ", "
             << half_ty << ") outs(" << theta_init << " : " << half_ty << ") -> " << half_ty << "\n";

        // cos_k, sin_k
        std::string cos_init = new_id();
        body << "    " << cos_init << " = tensor.empty() : " << half_ty << "\n";
        std::string cos_t = new_id();
        body << "    " << cos_t << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "
                "iterator_types = [\"parallel\"]} ins("
             << theta << " : " << half_ty << ") outs(" << cos_init << " : " << half_ty << ") {\n";
        body << "    ^bb0(%t: f32, %out: f32):\n";
        body << "      %c = math.cos %t : f32\n";
        body << "      linalg.yield %c : f32\n";
        body << "    } -> " << half_ty << "\n";

        std::string sin_init = new_id();
        body << "    " << sin_init << " = tensor.empty() : " << half_ty << "\n";
        std::string sin_t = new_id();
        body << "    " << sin_t << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "
                "iterator_types = [\"parallel\"]} ins("
             << theta << " : " << half_ty << ") outs(" << sin_init << " : " << half_ty << ") {\n";
        body << "    ^bb0(%t: f32, %out: f32):\n";
        body << "      %s = math.sin %t : f32\n";
        body << "      linalg.yield %s : f32\n";
        body << "    } -> " << half_ty << "\n";

        // deinterleave x into even/odd slices
        std::string x_even = new_id();
        body << "    " << x_even << " = tensor.extract_slice " << x_val << "[0, 0] [" << rows << ", " << half
             << "] [1, 2] : " << x_ty << " to " << pair_ty << "\n";
        std::string x_odd = new_id();
        body << "    " << x_odd << " = tensor.extract_slice " << x_val << "[0, 1] [" << rows << ", " << half
             << "] [1, 2] : " << x_ty << " to " << pair_ty << "\n";

        // rotate: out_even = x_even*cos - x_odd*sin; out_odd = x_even*sin + x_odd*cos
        std::string out_even_init = new_id();
        body << "    " << out_even_init << " = tensor.empty() : " << pair_ty << "\n";
        std::string out_odd_init = new_id();
        body << "    " << out_odd_init << " = tensor.empty() : " << pair_ty << "\n";
        std::string out_even = new_id();
        std::string out_odd  = new_id();
        body << "    " << out_even << ", " << out_odd << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>, "
                "affine_map<(d0,d1) -> (d1)>, affine_map<(d0,d1) -> (d1)>, "
                "affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], "
                "iterator_types = [\"parallel\", \"parallel\"]} ins("
             << x_even << ", " << x_odd << ", " << cos_t << ", " << sin_t << " : " << pair_ty << ", " << pair_ty
             << ", " << half_ty << ", " << half_ty << ") outs(" << out_even_init << ", " << out_odd_init << " : "
             << pair_ty << ", " << pair_ty << ") {\n";
        body << "    ^bb0(%xe: f32, %xo: f32, %c: f32, %s: f32, %oe: f32, %oo: f32):\n";
        body << "      %e1 = arith.mulf %xe, %c : f32\n";
        body << "      %e2 = arith.mulf %xo, %s : f32\n";
        body << "      %new_e = arith.subf %e1, %e2 : f32\n";
        body << "      %o1 = arith.mulf %xe, %s : f32\n";
        body << "      %o2 = arith.mulf %xo, %c : f32\n";
        body << "      %new_o = arith.addf %o1, %o2 : f32\n";
        body << "      linalg.yield %new_e, %new_o : f32, f32\n";
        body << "    } -> (" << pair_ty << ", " << pair_ty << ")\n";

        // re-interleave into the final result
        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";
        std::string out_with_even = new_id();
        body << "    " << out_with_even << " = tensor.insert_slice " << out_even << " into " << out_init << "[0, 0] ["
             << rows << ", " << half << "] [1, 2] : " << pair_ty << " into " << out_ty << "\n";
        std::string result = new_id();
        body << "    " << result << " = tensor.insert_slice " << out_odd << " into " << out_with_even << "[0, 1] ["
             << rows << ", " << half << "] [1, 2] : " << pair_ty << " into " << out_ty << "\n";

        return result;
    }

    // Rank-3 x = (head_dim, n_head, n_tokens) in ggml ne order -> MLIR shape (T, H, D) where
    // T=x->ne[2] (tokens, position varies here - the outermost/slowest MLIR dim), H=x->ne[1]
    // (heads, pure broadcast dim, no position dependence), D=x->ne[0] (head_dim, rotated).
    // pos must have exactly T elements (ggml's own pos->ne[0] == x->ne[2] constraint).
    std::string emit_rope_rank3(const ggml_tensor * node, const ggml_tensor * x, const ggml_tensor * pos,
                                 int32_t rot_dims, float freq_base) {
        int64_t T = x->ne[2];
        int64_t H = x->ne[1];
        int64_t half = rot_dims / 2;
        if (ggml_nelements(pos) != T) {
            fprintf(stderr, "mlir-export: ROPE (3D x) requires pos to have exactly x->ne[2] elements\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        const std::string & x_val   = values.at(x);
        const std::string & pos_val = values.at(pos);
        std::string          elem   = mlir_element_type(x);
        std::string          x_ty   = mlir_tensor_type(x);
        std::string          pos_ty = mlir_tensor_type(pos);
        std::string          pos_f32_ty = "tensor<" + std::to_string(T) + "x" + elem + ">";
        std::string          freq_ty    = "tensor<" + std::to_string(half) + "x" + elem + ">";
        std::string          theta_ty   = "tensor<" + std::to_string(T) + "x" + std::to_string(half) + "x" + elem + ">";
        std::string          pair_ty = "tensor<" + std::to_string(T) + "x" + std::to_string(H) + "x" +
                                        std::to_string(half) + "x" + elem + ">";

        // freq_k = freq_base^(-2k/n_dims), k=0..half-1 - compile-time constant. llama3 rope divides
        // each by freq_factors[k] (src[2]), matching ggml's theta/ff (Llama-3.x long-context scaling).
        const ggml_tensor * ff = node->src[2];
        const float * ffd = (ff && ff->data) ? (const float *) ff->data : nullptr;
        std::ostringstream freq_lit;
        freq_lit << "dense<[";
        for (int64_t k = 0; k < half; k++) {
            if (k > 0) {
                freq_lit << ", ";
            }
            float freq_k = powf(freq_base, -2.0f * (float) k / (float) rot_dims);
            if (ffd) { freq_k /= ffd[k]; }
            freq_lit << format_f32_literal(freq_k);
        }
        freq_lit << "]>";
        std::string freq_val = new_id();
        body << "    " << freq_val << " = arith.constant " << freq_lit.str() << " : " << freq_ty << "\n";

        // pos_f32[t] = sitofp(pos[t]) - elementwise convert over the whole position vector
        // (a genuine tensor operand, not a captured scalar - see emit_rope_rank2's comment on
        // why that distinction matters for the real pipeline's legalization).
        std::string pos_f32_init = new_id();
        body << "    " << pos_f32_init << " = tensor.empty() : " << pos_f32_ty << "\n";
        std::string pos_f32 = new_id();
        body << "    " << pos_f32 << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], "
                "iterator_types = [\"parallel\"]} ins("
             << pos_val << " : " << pos_ty << ") outs(" << pos_f32_init << " : " << pos_f32_ty << ") {\n";
        body << "    ^bb0(%in: i32, %out: f32):\n";
        body << "      %f = arith.sitofp %in : i32 to f32\n";
        body << "      linalg.yield %f : f32\n";
        body << "    } -> " << pos_f32_ty << "\n";

        // theta[t,k] = pos_f32[t] * freq[k] - an outer product built the same way as
        // rms_norm/soft_max's broadcast steps: each operand's own affine map selects which
        // of the (t,k) iteration dims it depends on.
        std::string theta_init = new_id();
        body << "    " << theta_init << " = tensor.empty() : " << theta_ty << "\n";
        std::string theta = new_id();
        body << "    " << theta << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0,d1) -> (d0)>, affine_map<(d0,d1) -> (d1)>, affine_map<(d0,d1) -> (d0,d1)>], "
                "iterator_types = [\"parallel\", \"parallel\"]} ins("
             << pos_f32 << ", " << freq_val << " : " << pos_f32_ty << ", " << freq_ty << ") outs(" << theta_init
             << " : " << theta_ty << ") {\n";
        body << "    ^bb0(%p: f32, %fr: f32, %out: f32):\n";
        body << "      %th = arith.mulf %p, %fr : f32\n";
        body << "      linalg.yield %th : f32\n";
        body << "    } -> " << theta_ty << "\n";

        // cos[t,k], sin[t,k]
        std::string cos_init = new_id();
        body << "    " << cos_init << " = tensor.empty() : " << theta_ty << "\n";
        std::string cos_t = new_id();
        body << "    " << cos_t << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], "
                "iterator_types = [\"parallel\", \"parallel\"]} ins("
             << theta << " : " << theta_ty << ") outs(" << cos_init << " : " << theta_ty << ") {\n";
        body << "    ^bb0(%t: f32, %out: f32):\n";
        body << "      %c = math.cos %t : f32\n";
        body << "      linalg.yield %c : f32\n";
        body << "    } -> " << theta_ty << "\n";

        std::string sin_init = new_id();
        body << "    " << sin_init << " = tensor.empty() : " << theta_ty << "\n";
        std::string sin_t = new_id();
        body << "    " << sin_t << " = linalg.generic {indexing_maps = ["
                "affine_map<(d0,d1) -> (d0,d1)>, affine_map<(d0,d1) -> (d0,d1)>], "
                "iterator_types = [\"parallel\", \"parallel\"]} ins("
             << theta << " : " << theta_ty << ") outs(" << sin_init << " : " << theta_ty << ") {\n";
        body << "    ^bb0(%t: f32, %out: f32):\n";
        body << "      %s = math.sin %t : f32\n";
        body << "      linalg.yield %s : f32\n";
        body << "    } -> " << theta_ty << "\n";

        // deinterleave x into even/odd slices along the innermost (head_dim) dim
        std::string x_even = new_id();
        body << "    " << x_even << " = tensor.extract_slice " << x_val << "[0, 0, 0] [" << T << ", " << H << ", "
             << half << "] [1, 1, 2] : " << x_ty << " to " << pair_ty << "\n";
        std::string x_odd = new_id();
        body << "    " << x_odd << " = tensor.extract_slice " << x_val << "[0, 0, 1] [" << T << ", " << H << ", "
             << half << "] [1, 1, 2] : " << x_ty << " to " << pair_ty << "\n";

        // rotate: broadcasting cos/sin (T,half) over the middle head dim H.
        std::string cos_sin_bcast_map = affine_map_select(3, {0, 2});
        std::string full3_map         = affine_map_full(3);
        std::string out_even_init = new_id();
        body << "    " << out_even_init << " = tensor.empty() : " << pair_ty << "\n";
        std::string out_odd_init = new_id();
        body << "    " << out_odd_init << " = tensor.empty() : " << pair_ty << "\n";
        std::string out_even = new_id();
        std::string out_odd  = new_id();
        body << "    " << out_even << ", " << out_odd << " = linalg.generic {indexing_maps = ["
             << full3_map << ", " << full3_map << ", " << cos_sin_bcast_map << ", " << cos_sin_bcast_map << ", "
             << full3_map << ", " << full3_map << "], "
                "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]} ins("
             << x_even << ", " << x_odd << ", " << cos_t << ", " << sin_t << " : " << pair_ty << ", " << pair_ty
             << ", " << theta_ty << ", " << theta_ty << ") outs(" << out_even_init << ", " << out_odd_init << " : "
             << pair_ty << ", " << pair_ty << ") {\n";
        body << "    ^bb0(%xe: f32, %xo: f32, %c: f32, %s: f32, %oe: f32, %oo: f32):\n";
        body << "      %e1 = arith.mulf %xe, %c : f32\n";
        body << "      %e2 = arith.mulf %xo, %s : f32\n";
        body << "      %new_e = arith.subf %e1, %e2 : f32\n";
        body << "      %o1 = arith.mulf %xe, %s : f32\n";
        body << "      %o2 = arith.mulf %xo, %c : f32\n";
        body << "      %new_o = arith.addf %o1, %o2 : f32\n";
        body << "      linalg.yield %new_e, %new_o : f32, f32\n";
        body << "    } -> (" << pair_ty << ", " << pair_ty << ")\n";

        // re-interleave into the final result
        std::string out_ty   = mlir_tensor_type(node);
        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";
        std::string out_with_even = new_id();
        body << "    " << out_with_even << " = tensor.insert_slice " << out_even << " into " << out_init
             << "[0, 0, 0] [" << T << ", " << H << ", " << half << "] [1, 1, 2] : " << pair_ty << " into " << out_ty
             << "\n";
        std::string result = new_id();
        body << "    " << result << " = tensor.insert_slice " << out_odd << " into " << out_with_even
             << "[0, 0, 1] [" << T << ", " << H << ", " << half << "] [1, 1, 2] : " << pair_ty << " into " << out_ty
             << "\n";

        return result;
    }

    // ggml_permute(a, axis0..axis3): result->ne[axis_i] = a->ne[i] - a pure axis relabeling
    // (a view in ggml's own lazy/strided model, with no data movement). Since every value in
    // our exported graph is already a fully materialized MLIR tensor (not a lazy view), the
    // equivalent here is a real linalg.transpose that performs the reorder. Only
    // rank-preserving permutations of rank 2 or 3 are supported (what real
    // attention needs: swapping the head/token dims before/after the batched QK^T and
    // scores*V matmuls) - promoting/demoting a trivial dim via permute is out of scope.
    //
    // Deriving the linalg.transpose permutation from ggml's axis array requires converting
    // between ggml's dim order and this file's MLIR dim order (MLIR reverses ggml's order,
    // see the top-of-file convention comment): for effective rank R, ggml dim g sits at MLIR
    // position R-1-g. If ggml moves dim g to position axis[g], the corresponding MLIR
    // transpose must place old MLIR position (R-1-g) at new MLIR position (R-1-axis[g]).
    // MLIR (reversed-ne) dims of a tensor over its effective rank.
    static std::vector<int64_t> mlir_dims(const ggml_tensor * t) {
        std::vector<int64_t> v;
        for (int i = ggml_n_dims(t) - 1; i >= 0; i--) {
            v.push_back(t->ne[i]);
        }
        return v;
    }

    // Reshape src (MLIR shape src_dims) to dst_dims when they differ only in where size-1 dims sit
    // (same non-1 dims, same order) - the n_tokens=1 case where a permute of the size-1 token dim
    // moves no data. Collapse to the dense (no size-1) shape, then expand to the target.
    std::string emit_size1_reshape(const std::string & src_val, const std::vector<int64_t> & src_dims,
                                   const std::vector<int64_t> & dst_dims, const std::string & elem) {
        auto ty = [&](const std::vector<int64_t> & s) {
            std::ostringstream o;
            o << "tensor<";
            for (int64_t d : s) o << d << "x";
            o << elem << ">";
            return o.str();
        };
        // Group dims so each group holds exactly one non-1 dim (size-1 dims attach to the preceding
        // non-1's group, leading 1s to the first) - the reassociation for collapse/expand vs dense.
        auto reassoc = [&](const std::vector<int64_t> & s) {
            std::vector<std::vector<int>> groups;
            bool has_nonone = false;
            for (int i = 0; i < (int) s.size(); i++) {
                if (s[i] != 1 && has_nonone) { groups.push_back({}); has_nonone = false; }
                if (groups.empty()) groups.push_back({});
                groups.back().push_back(i);
                if (s[i] != 1) has_nonone = true;
            }
            std::ostringstream o;
            o << "[";
            for (int g = 0; g < (int) groups.size(); g++) {
                if (g) o << ", ";
                o << "[";
                for (int j = 0; j < (int) groups[g].size(); j++) { if (j) o << ", "; o << groups[g][j]; }
                o << "]";
            }
            o << "]";
            return o.str();
        };
        std::vector<int64_t> dense;
        for (int64_t d : src_dims) if (d != 1) dense.push_back(d);

        std::string cur = src_val;
        if (src_dims.size() != dense.size()) {   // collapse src -> dense
            std::string c = new_id();
            body << "    " << c << " = tensor.collapse_shape " << cur << " " << reassoc(src_dims) << " : "
                 << ty(src_dims) << " into " << ty(dense) << "\n";
            cur = c;
        }
        if (dst_dims.size() != dense.size()) {   // expand dense -> dst
            std::string e = new_id();
            body << "    " << e << " = tensor.expand_shape " << cur << " " << reassoc(dst_dims) << " output_shape [";
            for (int i = 0; i < (int) dst_dims.size(); i++) { if (i) body << ", "; body << dst_dims[i]; }
            body << "] : " << ty(dense) << " into " << ty(dst_dims) << "\n";
            cur = e;
        }
        return cur;
    }

    std::string emit_permute(const ggml_tensor * node) {
        const ggml_tensor * x = node->src[0];
        int                 R = ggml_n_dims(x);

        // n_tokens=1: if the permute only reshuffles size-1 dims (relative order of the non-1 dims is
        // preserved), it moves no data - emit it as a reshape rather than a rank-preserving transpose.
        {
            const int32_t * axis4 = (const int32_t *) node->op_params;
            bool size1_only = true;
            for (int i = 0; i < 4 && size1_only; i++) {
                for (int j = i + 1; j < 4; j++) {
                    if (x->ne[i] > 1 && x->ne[j] > 1 && axis4[i] > axis4[j]) { size1_only = false; break; }
                }
            }
            if (size1_only) {
                return emit_size1_reshape(values.at(x), mlir_dims(x), mlir_dims(node), mlir_element_type(x));
            }
        }

        if (R < 2 || R > 3) {
            fprintf(stderr, "mlir-export: PERMUTE only supports rank 2 or 3 for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }
        if (ggml_n_dims(node) != R) {
            fprintf(stderr, "mlir-export: PERMUTE only supports rank-preserving permutations for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        const int32_t * axis = (const int32_t *) node->op_params;
        for (int g = 0; g < R; g++) {
            if (axis[g] < 0 || axis[g] >= R) {
                fprintf(stderr,
                        "mlir-export: PERMUTE only supports permutations confined to the effective rank (no "
                        "promoting/demoting trivial dims) for now\n");
                throw mlir_export_error("unsupported graph construct (see message above)");
            }
        }

        std::vector<int> perm(R);
        for (int g = 0; g < R; g++) {
            int m  = R - 1 - g;
            int mp = R - 1 - (int) axis[g];
            perm[mp] = m;
        }

        const std::string & x_val = values.at(x);
        std::string          x_ty   = mlir_tensor_type(x);
        std::string          out_ty = mlir_tensor_type(node);

        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";
        std::string result = new_id();
        body << "    " << result << " = linalg.transpose ins(" << x_val << " : " << x_ty << ") outs(" << out_init
             << " : " << out_ty << ") permutation = [";
        for (int i = 0; i < R; i++) {
            if (i > 0) {
                body << ", ";
            }
            body << perm[i];
        }
        body << "]\n";

        return result;
    }

    // Shared by GGML_OP_RESHAPE and GGML_OP_CONT: both reinterpret the same total element
    // count under a new logical shape. GGML_OP_CONT's ggml_cont_2d/3d/4d variants fuse
    // "materialize a (possibly permuted/non-contiguous) view" with "reshape to a new shape"
    // into one node - but since our exporter already materializes every value eagerly (the
    // preceding PERMUTE already emitted a real linalg.transpose), by the time a CONT node is
    // reached its source value is already exactly what ggml's own cont would produce, so
    // "materialize" is a no-op here and only the shape change (if any) needs emitting.
    //
    // Only two shape changes are supported, both matching what real multi-head attention
    // needs: splitting a 2D tensor's innermost ggml dim into (head_dim, n_head) before
    // per-head ops (tensor.expand_shape), and merging a 3D tensor's outer two ggml dims back
    // into one after them (tensor.collapse_shape) - plus a same-shape passthrough for a plain
    // ggml_cont() with no reshape.
    std::string emit_reshape_like(const ggml_tensor * node, const ggml_tensor * x) {
        int x_dims = ggml_n_dims(x);
        int out_dims = ggml_n_dims(node);

        const std::string & x_val = values.at(x);

        bool same_shape = x_dims == out_dims;
        for (int i = 0; same_shape && i < x_dims; i++) {
            same_shape = x->ne[i] == node->ne[i];
        }
        if (same_shape) {
            return x_val;
        }

        std::string x_ty   = mlir_tensor_type(x);
        std::string out_ty = mlir_tensor_type(node);

        if (x_dims == 2 && out_dims == 3 && node->ne[2] == x->ne[1] && node->ne[0] * node->ne[1] == x->ne[0]) {
            std::string result = new_id();
            body << "    " << result << " = tensor.expand_shape " << x_val << " [[0], [1, 2]] output_shape ["
                 << node->ne[2] << ", " << node->ne[1] << ", " << node->ne[0] << "] : " << x_ty << " into " << out_ty
                 << "\n";
            return result;
        }
        if (x_dims == 3 && out_dims == 2 && x->ne[2] == node->ne[1] && x->ne[0] * x->ne[1] == node->ne[0]) {
            std::string result = new_id();
            body << "    " << result << " = tensor.collapse_shape " << x_val << " [[0], [1, 2]] : " << x_ty
                 << " into " << out_ty << "\n";
            return result;
        }
        // n_tokens=1: 1D hidden -> 2D heads (q/k/v proj -> heads), e.g. [hidden,1] -> [head_dim,n_head,1]
        if (x_dims == 1 && out_dims == 2 && node->ne[0] * node->ne[1] == x->ne[0]) {
            std::string result = new_id();
            body << "    " << result << " = tensor.expand_shape " << x_val << " [[0, 1]] output_shape ["
                 << node->ne[1] << ", " << node->ne[0] << "] : " << x_ty << " into " << out_ty << "\n";
            return result;
        }
        // n_tokens=1: 2D heads -> 1D hidden (head merge), e.g. [head_dim,n_head,1] -> [hidden,1]
        if (x_dims == 2 && out_dims == 1 && x->ne[0] * x->ne[1] == node->ne[0]) {
            std::string result = new_id();
            body << "    " << result << " = tensor.collapse_shape " << x_val << " [[0, 1]] : " << x_ty
                 << " into " << out_ty << "\n";
            return result;
        }

        fprintf(stderr,
                "mlir-export: RESHAPE/CONT only supports a same-shape passthrough, a 2D->3D head split, or a "
                "3D->2D head merge for now\n");
        throw mlir_export_error("unsupported graph construct (see message above)");
    }

    // ggml_concat(a, b, dim): joins a and b along ggml dim `dim` (op_params[0]). Used by the KV-cache
    // decode graph to append the new token's K/V to the cache. ggml dim d maps to MLIR dim R-1-d
    // (reversed ne), and ggml's [a; b] order is the MLIR concat operand order. An operand can report
    // fewer ggml dims than the node (e.g. new K/V [.,.,1] vs cache [.,.,cur]); expand it to rank R.
    std::string emit_concat(const ggml_tensor * node) {
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        const int R = ggml_n_dims(node);
        const int32_t dim = ((const int32_t *) node->op_params)[0];
        if (dim < 0 || dim >= R) {
            fprintf(stderr, "mlir-export: CONCAT dim %d out of range for rank %d\n", (int) dim, R);
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        // bring an operand up to rank R via tensor.expand_shape (ggml trailing-1 -> MLIR leading-1)
        auto to_rankR = [&](const ggml_tensor * op) -> std::string {
            const int r = ggml_n_dims(op);
            if (r == R) return values.at(op);
            std::string dst = new_id();
            std::ostringstream re;                       // reassociation [[0..R-r], [R-r+1], ..., [R-1]]
            re << "[[";
            for (int i = 0; i <= R - r; i++) { if (i) re << ", "; re << i; }
            re << "]";
            for (int i = R - r + 1; i < R; i++) re << ", [" << i << "]";
            re << "]";
            std::ostringstream osh;                      // output_shape = MLIR dims (reversed ne over R)
            osh << "[";
            for (int i = R - 1; i >= 0; i--) { if (i != R - 1) osh << ", "; osh << op->ne[i]; }
            osh << "]";
            body << "    " << dst << " = tensor.expand_shape " << values.at(op) << " " << re.str()
                 << " output_shape " << osh.str()
                 << " : " << mlir_tensor_type(op) << " into " << mlir_tensor_type_ranked(op, R) << "\n";
            return dst;
        };
        std::string av = to_rankR(a), bv = to_rankR(b);

        // Build the concat from tensor.empty + insert_slice (tensor.concat isn't bufferized by the
        // TSI pipeline). `a` fills [0 : a.dim) along the concat dim; `b` fills [a.dim : a.dim+b.dim).
        const int        mlir_dim = R - 1 - dim;
        const std::string out_ty  = mlir_tensor_type(node);
        auto dims = [&](const ggml_tensor * t) {           // MLIR sizes "[d0, d1, ...]" at rank R
            std::ostringstream o; o << "[";
            for (int j = 0; j < R; j++) { if (j) o << ", "; o << t->ne[R - 1 - j]; }
            o << "]"; return o.str();
        };
        std::ostringstream zeros, ones, boff;
        zeros << "["; ones << "["; boff << "[";
        for (int j = 0; j < R; j++) {
            if (j) { zeros << ", "; ones << ", "; boff << ", "; }
            zeros << "0"; ones << "1";
            boff << (j == mlir_dim ? (long long) a->ne[dim] : 0);   // b starts after a on the concat dim
        }
        zeros << "]"; ones << "]"; boff << "]";

        std::string init = new_id();
        body << "    " << init << " = tensor.empty() : " << out_ty << "\n";
        std::string r0 = new_id();
        body << "    " << r0 << " = tensor.insert_slice " << av << " into " << init << zeros.str() << " "
             << dims(a) << " " << ones.str() << " : " << mlir_tensor_type_ranked(a, R) << " into " << out_ty << "\n";
        std::string r1 = new_id();
        body << "    " << r1 << " = tensor.insert_slice " << bv << " into " << r0 << boff.str() << " "
             << dims(b) << " " << ones.str() << " : " << mlir_tensor_type_ranked(b, R) << " into " << out_ty << "\n";
        return r1;
    }

    // ggml_get_rows(a, b): gathers rows of embedding table `a` (ggml ne=(n_embd, n_vocab))
    // by integer indices in token-id vector `b` (ggml ne=(n_tokens,), I32) - real llama.cpp's
    // token embedding lookup. Scope: 2D a, 1D b only (the real single-sequence case; ggml's
    // own ggml_get_rows precondition collapses a->ne[2]/b->ne[1] and a->ne[3]/b->ne[2] to 1
    // for this shape anyway).
    //
    // Unlike every other slice operation in this file (RoPE's deinterleave, the GQA head
    // repeat), token ids are genuine runtime data - an actual function argument, not baked
    // into the exported IR at compile time - so each gathered row needs a DYNAMIC-offset
    // tensor.extract_slice rather than a compile-time-constant one. The token *count* is
    // still compile-time known, so this unrolls one extract+insert pair per token, the same
    // shape of loop as emit_repeat_heads_3d's per-head unrolling.
    std::string emit_get_rows(const ggml_tensor * node) {
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];

        if (ggml_n_dims(a) != 2 || ggml_n_dims(b) != 1) {
            fprintf(stderr,
                    "mlir-export: GET_ROWS only supports a 2D embedding table and a 1D token-id vector for now\n");
            throw mlir_export_error("unsupported graph construct (see message above)");
        }

        int64_t n_embd   = a->ne[0];
        int64_t n_tokens = b->ne[0];

        const std::string & a_val = values.at(a);
        const std::string & b_val = values.at(b);
        std::string          a_ty   = mlir_tensor_type(a);
        std::string          b_ty   = mlir_tensor_type(b);
        std::string          row_ty = "tensor<1x" + std::to_string(n_embd) + "x" + mlir_element_type(a) + ">";
        std::string          out_ty = mlir_tensor_type(node);

        // n_tokens=1 (decode): output collapses to rank-1 [n_embd]; extract the single row directly
        // with a rank-reducing slice (the [1, n_embd] slice into a rank-1 result drops the unit dim).
        if (n_tokens == 1 && ggml_n_dims(node) == 1) {
            std::string c_t = new_id();
            body << "    " << c_t << " = arith.constant 0 : index\n";
            std::string tok_i32 = new_id();
            body << "    " << tok_i32 << " = tensor.extract " << b_val << "[" << c_t << "] : " << b_ty << "\n";
            std::string tok_idx = new_id();
            body << "    " << tok_idx << " = arith.index_cast " << tok_i32 << " : i32 to index\n";
            std::string row = new_id();
            body << "    " << row << " = tensor.extract_slice " << a_val << "[" << tok_idx << ", 0] [1, " << n_embd
                 << "] [1, 1] : " << a_ty << " to " << out_ty << "\n";
            return row;
        }

        std::string out_init = new_id();
        body << "    " << out_init << " = tensor.empty() : " << out_ty << "\n";

        std::string current = out_init;
        for (int64_t t = 0; t < n_tokens; t++) {
            std::string c_t = new_id();
            body << "    " << c_t << " = arith.constant " << t << " : index\n";
            std::string tok_i32 = new_id();
            body << "    " << tok_i32 << " = tensor.extract " << b_val << "[" << c_t << "] : " << b_ty << "\n";
            std::string tok_idx = new_id();
            body << "    " << tok_idx << " = arith.index_cast " << tok_i32 << " : i32 to index\n";
            std::string row = new_id();
            body << "    " << row << " = tensor.extract_slice " << a_val << "[" << tok_idx << ", 0] [1, " << n_embd
                 << "] [1, 1] : " << a_ty << " to " << row_ty << "\n";
            std::string next = new_id();
            body << "    " << next << " = tensor.insert_slice " << row << " into " << current << "[" << t
                 << ", 0] [1, " << n_embd << "] [1, 1] : " << row_ty << " into " << out_ty << "\n";
            current = next;
        }

        return current;
    }
};

// discover leaf/input tensors in first-seen order, without touching ggml_cgraph internals
// (leafs[]/n_leafs are private - src[] on the public ggml_tensor is enough).
static std::vector<const ggml_tensor *> discover_leafs(struct ggml_cgraph * gf) {
    std::vector<const ggml_tensor *>   leafs;
    std::map<const ggml_tensor *, int> leaf_index;
    int                                n_nodes = ggml_graph_n_nodes(gf);

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            struct ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                continue;
            }
            if (src->op == GGML_OP_NONE && leaf_index.find(src) == leaf_index.end()) {
                leaf_index[src] = (int) leafs.size();
                leafs.push_back(src);
            }
        }
    }
    return leafs;
}

// builds just the "func.func @name(...) { ... }" block (no module wrapper), so multiple
// cases can be combined into a single module.
// dispatches every node in `gf` to the matching emit_* function, in the graph's own
// (already topologically-sorted) order. Independent of which leafs end up as function
// parameters vs. baked-in constants, since that distinction only affects how leafs
// themselves are seeded into `ex.values` before this runs.
static void dispatch_graph_nodes(struct ggml_cgraph * gf, linalg_exporter & ex, struct ggml_tensor *& out) {
    int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);

        std::string val;
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                val = ex.emit_mul_mat(node);
                break;
            case GGML_OP_ADD:
                val = ex.emit_add(node);
                break;
            case GGML_OP_RMS_NORM:
                val = ex.emit_rms_norm(node);
                break;
            case GGML_OP_MUL:
                val = ex.emit_mul(node);
                break;
            case GGML_OP_SCALE:
                val = ex.emit_scale(node);
                break;
            case GGML_OP_SOFT_MAX:
                val = ex.emit_soft_max(node);
                break;
            case GGML_OP_ROPE:
                val = ex.emit_rope(node);
                break;
            case GGML_OP_PERMUTE:
                val = ex.emit_permute(node);
                break;
            case GGML_OP_RESHAPE:
            case GGML_OP_CONT:
                val = ex.emit_reshape_like(node, node->src[0]);
                break;
            case GGML_OP_GET_ROWS:
                val = ex.emit_get_rows(node);
                break;
            case GGML_OP_CONCAT:
                val = ex.emit_concat(node);
                break;
            case GGML_OP_UNARY:
                if (ggml_get_unary_op(node) == GGML_UNARY_OP_SILU) {
                    val = ex.emit_silu(node);
                } else {
                    fprintf(stderr, "mlir-export: unsupported unary op: %s\n",
                            ggml_unary_op_name(ggml_get_unary_op(node)));
                    throw mlir_export_error("unsupported graph construct (see message above)");
                }
                break;
            default:
                fprintf(stderr, "mlir-export: unsupported op: %s\n", ggml_op_name(node->op));
                throw mlir_export_error("unsupported graph construct (see message above)");
        }

        ex.values[node] = val;
        out             = node;
    }
}

// recursive helper for mlir_dense_literal: emits nested "[...]" brackets for ggml dim `dim`
// downward (dim counts down from n_dims-1 to 0), using t's REAL byte strides (t->nb[]) to
// compute each element's byte offset from `base_byte_offset` - required for correctness on
// a non-contiguous view (GGML_OP_PERMUTE's direct output aliases its source tensor's buffer
// with reordered strides, not a freshly packed layout;
// assuming element stride == product of smaller ne's silently reads the wrong values for
// that case). Walking ggml dims from n_dims-1 (outermost/slowest) down to 0
// (innermost/fastest) produces exactly the nesting order MLIR's dense literal expects, since
// MLIR's tensor shape is (ne[n_dims-1], ..., ne[0]) - the reverse of ggml's own dim order.
static void mlir_dense_literal_dim(const ggml_tensor * t, int dim, size_t base_byte_offset,
                                    std::ostringstream & oss) {
    auto format_elem = [&](size_t byte_offset) {
        const void * p = (const char *) t->data + byte_offset;
        if (t->type == GGML_TYPE_I32) {
            oss << *(const int32_t *) p;
        } else {
            oss << format_f32_literal(*(const float *) p);
        }
    };

    int64_t n = t->ne[dim];
    oss << "[";
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) {
            oss << ", ";
        }
        size_t byte_offset = base_byte_offset + (size_t) i * t->nb[dim];
        if (dim == 0) {
            format_elem(byte_offset);
        } else {
            mlir_dense_literal_dim(t, dim - 1, byte_offset, oss);
        }
    }
    oss << "]";
}

// dense elements attribute literal for an N-D tensor (f32 or i32), e.g.
// "dense<[[1.0, 2.0], [3.0, 4.0]]>" (2D f32) or "dense<[3]>" (1D i32).
static std::string mlir_dense_literal(const ggml_tensor * t) {
    int n_dims = ggml_n_dims(t);

    std::ostringstream oss;
    oss << "dense<";
    mlir_dense_literal_dim(t, n_dims - 1, 0, oss);
    oss << ">";
    return oss.str();
}

// Emit the graph as one func @forward. `const_leafs` are baked into the body as arith.constant
// values (via mlir_dense_literal); only `runtime_args` become %argN parameters. Baking the weights
// keeps the signature small for a many-layer graph, leaving genuinely-runtime data as arguments.
static std::string build_func_text_baked(struct ggml_cgraph * gf, const char * func_name,
                                          const std::vector<const ggml_tensor *> & runtime_args,
                                          const std::vector<const ggml_tensor *> & const_leafs) {
    int n_nodes = ggml_graph_n_nodes(gf);
    if (n_nodes == 0) {
        fprintf(stderr, "mlir-export: graph has no nodes\n");
        throw mlir_export_error("unsupported graph construct (see message above)");
    }
    if (runtime_args.empty()) {
        fprintf(stderr, "mlir-export: graph has no runtime input tensors\n");
        throw mlir_export_error("unsupported graph construct (see message above)");
    }

    linalg_exporter ex;

    std::vector<std::string> arg_decls;
    for (size_t i = 0; i < runtime_args.size(); i++) {
        std::string arg          = "%arg" + std::to_string(i);
        ex.values[runtime_args[i]] = arg;

        std::ostringstream decl;
        decl << arg << ": " << mlir_tensor_type(runtime_args[i]) << " {txe.name = \"input_" << i << "\"}";
        arg_decls.push_back(decl.str());
    }

    for (const ggml_tensor * leaf : const_leafs) {
        std::string id = ex.new_id();
        ex.body << "    " << id << " = arith.constant " << mlir_dense_literal(leaf) << " : " << mlir_tensor_type(leaf)
                << "\n";
        ex.values[leaf] = id;
    }

    struct ggml_tensor * out = nullptr;
    dispatch_graph_nodes(gf, ex, out);

    std::ostringstream f;
    f << "  func.func @" << func_name << "(";
    for (size_t i = 0; i < arg_decls.size(); i++) {
        if (i > 0) {
            f << ", ";
        }
        f << arg_decls[i];
    }
    f << ") -> (" << mlir_tensor_type(out) << " {txe.name = \"res_0\"}) attributes {llvm.emit_c_interface} {\n";
    f << ex.body.str();
    f << "    return " << ex.values.at(out) << " : " << mlir_tensor_type(out) << "\n";
    f << "  }\n";

    return f.str();
}

// Multi-output variant: returns the given `outputs` in order (e.g. logits + per-layer k_new/v_new for
// the KV-cache decode graph). MLIR's emit-c-interface appends the result out-params after the inputs,
// so the ciface arg order is [runtime_args..., outputs...] (verified against the single-output case).
static std::string build_func_text_baked_multi(struct ggml_cgraph * gf, const char * func_name,
                                               const std::vector<const ggml_tensor *> & runtime_args,
                                               const std::vector<const ggml_tensor *> & const_leafs,
                                               const std::vector<const ggml_tensor *> & outputs) {
    if (ggml_graph_n_nodes(gf) == 0) { fprintf(stderr, "mlir-export: graph has no nodes\n"); throw mlir_export_error("unsupported graph construct (see message above)"); }
    if (runtime_args.empty())        { fprintf(stderr, "mlir-export: graph has no runtime input tensors\n"); throw mlir_export_error("unsupported graph construct (see message above)"); }
    if (outputs.empty())             { fprintf(stderr, "mlir-export: no outputs given\n"); throw mlir_export_error("unsupported graph construct (see message above)"); }

    linalg_exporter ex;
    std::vector<std::string> arg_decls;
    for (size_t i = 0; i < runtime_args.size(); i++) {
        std::string arg = "%arg" + std::to_string(i);
        ex.values[runtime_args[i]] = arg;
        std::ostringstream decl; decl << arg << ": " << mlir_tensor_type(runtime_args[i]) << " {txe.name = \"input_" << i << "\"}";
        arg_decls.push_back(decl.str());
    }
    for (const ggml_tensor * leaf : const_leafs) {
        std::string id = ex.new_id();
        ex.body << "    " << id << " = arith.constant " << mlir_dense_literal(leaf) << " : " << mlir_tensor_type(leaf) << "\n";
        ex.values[leaf] = id;
    }
    struct ggml_tensor * ignored = nullptr;
    dispatch_graph_nodes(gf, ex, ignored);

    std::ostringstream f;
    f << "  func.func @" << func_name << "(";
    for (size_t i = 0; i < arg_decls.size(); i++) { if (i) f << ", "; f << arg_decls[i]; }
    f << ") -> (";
    for (size_t i = 0; i < outputs.size(); i++) { if (i) f << ", "; f << mlir_tensor_type(outputs[i]) << " {txe.name = \"res_" << i << "\"}"; }
    f << ") attributes {llvm.emit_c_interface} {\n";
    f << ex.body.str();
    f << "    return ";
    for (size_t i = 0; i < outputs.size(); i++) { if (i) f << ", "; f << ex.values.at(outputs[i]); }
    f << " : ";
    for (size_t i = 0; i < outputs.size(); i++) { if (i) f << ", "; f << mlir_tensor_type(outputs[i]); }
    f << "\n  }\n";
    return f.str();
}

struct case_result {
    struct ggml_context             * ctx;
    struct ggml_cgraph              * gf;
    std::vector<const ggml_tensor *>  leafs;         // every leaf tensor in the graph
    std::vector<const ggml_tensor *>  runtime_args;   // subset of `leafs` that become func.func
                                                       // %argN parameters (see build_func_text_baked)
    std::string                       func_text;
};
