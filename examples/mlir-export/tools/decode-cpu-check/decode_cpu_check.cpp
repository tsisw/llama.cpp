// CPU check for the fixed-L KV-cache decode graph (no FPGA). For each step it decodes one token with
// a host-side cache (padded to L, unfilled slots masked) and checks the argmax equals a full prefill
// of the same prefix: decode step k (cache = tokens 0..k-1) == prefill(0..k) last position.
// --emit writes the decode graph as one multi-output MLIR func (logits + per-layer k_new/v_new).
//
// Usage: decode_cpu_check <model.gguf> <id0> [id1 ...] [--L N] [--emit forward.mlir]
#include "tsi/graph/DecodeModel.h"   // load_decode_model, build_decode (pulls decode_layer.h -> model_layer.h)
#include "tsi/export/Exporter.h"       // exportGraph, discoverLeafs
#include "ggml-cpu.h"

#include <fstream>
#include <string>
#include <vector>

static int argmax(const float * v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf id0 [id1 ...] [--L N] [--emit file]\n", argv[0]); return 1; }
    const char * path = argv[1];
    std::vector<int32_t> ids;
    int L_arg = -1; const char * emit = nullptr;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--L") && i + 1 < argc)         L_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit = argv[++i];
        else ids.push_back(atoi(argv[i]));
    }
    const int N = (int) ids.size();

    DecodeModel M = load_decode_model(path);
    const int L = L_arg > 0 ? L_arg : (N + 4);   // padded cache length (extra slots exercise the mask)
    const int VOC = M.n_vocab, KV = M.hidden_kv;
    fprintf(stderr, "dims: layers=%d hidden=%d n_head=%d/%d head_dim=%d vocab=%d L=%d ntok=%d\n",
            M.n_layers, M.hidden, M.n_head, M.n_head_kv, M.head_dim, VOC, L, N);

    // emit the fixed-L decode graph as one multi-output MLIR func
    if (emit) {
        ggml_init_params ep { (size_t) 128 << 20, nullptr, false };
        ggml_context * ec = ggml_init(ep);
        ggml_tensor * id   = ggml_new_tensor_1d(ec, GGML_TYPE_I32, 1);
        ggml_tensor * pos  = ggml_new_tensor_1d(ec, GGML_TYPE_I32, 1);
        ggml_tensor * mask = ggml_new_tensor_2d(ec, GGML_TYPE_F32, L + 1, 1);
        std::vector<ggml_tensor *> cK(M.n_layers), cV(M.n_layers);
        for (int il = 0; il < M.n_layers; il++) {
            cK[il] = ggml_new_tensor_3d(ec, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L);
            cV[il] = ggml_new_tensor_3d(ec, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L);
        }
        std::vector<ggml_tensor *> knew, vnew;
        ggml_tensor * logits = build_decode(ec, M, id, pos, mask, cK, cV, knew, vnew);
        ggml_cgraph * gf = ggml_new_graph_custom(ec, 16384, false);
        ggml_build_forward_expand(gf, logits);
        for (int il = 0; il < M.n_layers; il++) { ggml_build_forward_expand(gf, knew[il]); ggml_build_forward_expand(gf, vnew[il]); }
        std::vector<const ggml_tensor *> outs; outs.push_back(logits);
        for (int il = 0; il < M.n_layers; il++) { outs.push_back(knew[il]); outs.push_back(vnew[il]); }
        auto leafs = tsi::mlir_export::discoverLeafs(gf);
        tsi::mlir_export::ExportOptions opts;
        opts.runtime_args = leafs;
        opts.outputs      = outs;
        // exportGraph returns a complete module, so no wrapping here any more.
        std::ofstream f(emit); f << tsi::mlir_export::exportGraph(gf, opts);
        fprintf(stderr, "emitted decode graph: L=%d leafs=%zu outputs=%zu -> %s\n", L, leafs.size(), outs.size(), emit);
        ggml_free(ec);
    }

    // decode-with-cache (fixed L + mask) vs prefill, per step
    std::vector<std::vector<float>> cacheK(M.n_layers, std::vector<float>((size_t) L * KV, 0.0f));
    std::vector<std::vector<float>> cacheV(M.n_layers, std::vector<float>((size_t) L * KV, 0.0f));
    int mism = 0;
    for (int k = 0; k < N; k++) {
        ggml_init_params dp { (size_t) 256 << 20, nullptr, false };
        ggml_context * dc = ggml_init(dp);
        ggml_tensor * id  = ggml_new_tensor_1d(dc, GGML_TYPE_I32, 1); ((int32_t *) id->data)[0]  = ids[k];
        ggml_tensor * pos = ggml_new_tensor_1d(dc, GGML_TYPE_I32, 1); ((int32_t *) pos->data)[0] = k;
        ggml_tensor * mask = ggml_new_tensor_2d(dc, GGML_TYPE_F32, L + 1, 1);
        { float * mm = (float *) mask->data;
          for (int j = 0; j < L; j++) mm[j] = (j < k) ? 0.0f : -INFINITY;   // valid cache [0,k), padded rest
          mm[L] = 0.0f; }                                                   // the new token
        std::vector<ggml_tensor *> cK(M.n_layers), cV(M.n_layers);
        for (int il = 0; il < M.n_layers; il++) {
            cK[il] = ggml_new_tensor_3d(dc, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L); memcpy(cK[il]->data, cacheK[il].data(), (size_t) L * KV * sizeof(float));
            cV[il] = ggml_new_tensor_3d(dc, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L); memcpy(cV[il]->data, cacheV[il].data(), (size_t) L * KV * sizeof(float));
        }
        std::vector<ggml_tensor *> knew, vnew;
        ggml_tensor * logits_d = build_decode(dc, M, id, pos, mask, cK, cV, knew, vnew);
        ggml_cgraph * gd = ggml_new_graph_custom(dc, 16384, false);
        ggml_build_forward_expand(gd, logits_d);
        for (int il = 0; il < M.n_layers; il++) { ggml_build_forward_expand(gd, knew[il]); ggml_build_forward_expand(gd, vnew[il]); }
        ggml_graph_compute_with_ctx(dc, gd, 4);

        std::vector<float> ld((size_t) VOC); memcpy(ld.data(), logits_d->data, (size_t) VOC * sizeof(float));
        int am_d = argmax(ld.data(), VOC);
        for (int il = 0; il < M.n_layers; il++) {          // append this token's K/V at slot k
            memcpy(cacheK[il].data() + (size_t) k * KV, knew[il]->data, KV * sizeof(float));
            memcpy(cacheV[il].data() + (size_t) k * KV, vnew[il]->data, KV * sizeof(float));
        }
        ggml_free(dc);

        // prefill reference (tokens 0..k, last position)
        const int m = k + 1;
        ggml_init_params pp { (size_t) 256 << 20, nullptr, false };
        ggml_context * pc = ggml_init(pp);
        ggml_tensor * idt = ggml_new_tensor_1d(pc, GGML_TYPE_I32, m); memcpy(idt->data, ids.data(), (size_t) m * sizeof(int32_t));
        ggml_tensor * pos2 = ggml_new_tensor_1d(pc, GGML_TYPE_I32, m); for (int i = 0; i < m; i++) ((int32_t *) pos2->data)[i] = i;
        ggml_tensor * pmask = ggml_new_tensor_2d(pc, GGML_TYPE_F32, m, m);
        { float * mm = (float *) pmask->data; for (int q = 0; q < m; q++) for (int j = 0; j < m; j++) mm[q * m + j] = (j <= q) ? 0.f : -INFINITY; }
        ggml_tensor * cp = ggml_get_rows(pc, M.embd, idt);
        for (int il = 0; il < M.n_layers; il++) cp = build_layer(pc, cp, M.lw[il], pos2, pmask, M.head_dim, M.n_head, M.n_head_kv, m);
        ggml_tensor * logits_p = ggml_mul_mat(pc, M.oww, ggml_mul(pc, ggml_rms_norm(pc, cp, REAL_RMS_EPS), M.onw));
        ggml_cgraph * gpf = ggml_new_graph_custom(pc, 16384, false);
        ggml_build_forward_expand(gpf, logits_p);
        ggml_graph_compute_with_ctx(pc, gpf, 4);
        const float * lp = (const float *) logits_p->data + (size_t) (m - 1) * VOC;
        int am_p = argmax(lp, VOC);

        double num = 0, den = 0; for (int v = 0; v < VOC; v++) { double dd = (double) ld[v] - lp[v]; num += dd * dd; den += (double) lp[v] * lp[v]; }
        bool ok = (am_d == am_p); if (!ok) mism++;
        fprintf(stderr, "step %2d (cur=%2d): decode=%d prefill=%d rel_sq_err=%.3g -> %s\n", k, k, am_d, am_p, den > 0 ? num / den : num, ok ? "MATCH" : "DIFFER");
        ggml_free(pc);
    }
    ggml_free(M.wc);
    fprintf(stderr, "=== fixed-L(%d) decode-vs-prefill: %d/%d MATCH ===\n", L, N - mism, N);
    return mism == 0 ? 0 : 1;
}
