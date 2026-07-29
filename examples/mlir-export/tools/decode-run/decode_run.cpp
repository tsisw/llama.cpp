// Stateful runner for the compiled fixed-L KV-cache decode graph. Loads the decode host.so (one
// @forward: logits + per-layer k_new/v_new) and drives it with a host-side cache: each step decodes
// one token at position cur, appends the returned k_new/v_new into the cache at slot cur, feeds the
// prediction back. Weights go to the device once; id/pos/mask/cache are refreshed per step. --verify
// diffs each token's argmax against a CPU prefill of the prefix. Box-only (needs the TSI runtime).
//
// Usage: decode_run <model.gguf> {--lib host.so | --emit forward.mlir} {--prompt "text" | id0 [ids...]} [--L N] [--gen N] [--verify]
#include "tsi/graph/DecodeModel.h"          // load_decode_model, build_decode, DecodeModel
#include "tsi/export/TextEmitter.h"              // discover_leafs
#include "include/TestModel.h"     // MemRefDescriptor<N>, tsi_alloc, tsi_dealloc
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-tsavorite.h"        // ggml_backend_tsavorite_init/_free (clean TSI runtime finalize)
#include "llama.h"                 // tokenize a --prompt, detokenize the output

#include <dlfcn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// heap MemRefDescriptor<N> over device pointer p (rank = ggml_n_dims, MLIR shape = ne reversed) -
// identical to the backend's make_desc so the compiled forward sees the same ABI.
template <int N> static void * make_desc_n(const ggml_tensor * t, void * p) {
    auto * d = new MemRefDescriptor<N>();
    d->base = p; d->data = p; d->offset = 0;
    for (int i = 0; i < N; i++) d->shape[i] = t->ne[N - 1 - i];
    d->strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) d->strides[i] = d->strides[i + 1] * d->shape[i + 1];
    return d;
}
static void * make_desc(const ggml_tensor * t, void * p) {
    switch (ggml_n_dims(t)) {
        case 1:  return make_desc_n<1>(t, p);
        case 2:  return make_desc_n<2>(t, p);
        case 3:  return make_desc_n<3>(t, p);
        default: return make_desc_n<4>(t, p);
    }
}
static int argmax(const float * v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

// drop llama's info/warn chatter (model-loader dump, print_info); keep errors
static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    llama_log_set(quiet_log, nullptr);
    const char * path = argc > 1 ? argv[1] : nullptr;
    const char * lib = nullptr, * emit = nullptr, * prompt = nullptr;
    std::vector<int32_t> ids;
    int L_arg = -1, gen = 0, verify = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--lib") && i + 1 < argc)     lib = argv[++i];
        else if (!strcmp(argv[i], "--emit") && i + 1 < argc) emit = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--L") && i + 1 < argc)  L_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify"))             verify = 1;
        else ids.push_back(atoi(argv[i]));
    }
    if (!path || (!lib && !emit) || (ids.empty() && !prompt)) {
        fprintf(stderr, "usage: %s model.gguf {--lib host.so | --emit forward.mlir} "
                        "{--prompt \"text\" | id0 [id1 ...]} [--L N] [--gen N] [--verify]\n", argv[0]);
        return 1;
    }

    // vocab_only load: tokenize --prompt / detokenize the output, no weights. llama_backend_init also
    // brings the TSI runtime up (via backend registration); ggml_backend_free finalizes it at exit.
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true; mp.n_gpu_layers = 0;
    llama_model * lmodel = llama_model_load_from_file(path, mp);
    if (!lmodel) { fprintf(stderr, "failed to load vocab from %s\n", path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lmodel);

    if (prompt) {
        std::vector<llama_token> toks((int) strlen(prompt) + 16);
        int n = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), (int) toks.size(), true, false);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt, (int) strlen(prompt), toks.data(), (int) toks.size(), true, false); }
        ids.assign(toks.begin(), toks.begin() + (n > 0 ? n : 0));
        printf("[decode] prompt \"%s\" -> %d tokens\n", prompt, (int) ids.size());
    }
    if (ids.empty()) { fprintf(stderr, "no prompt tokens\n"); return 1; }
    const int N = (int) ids.size();

    // load the compiled decode forward (skipped when only emitting)
    typedef void (*forward_argv_fn)(void **);
    forward_argv_fn fwd = nullptr;
    if (lib) {
        void * h = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
        if (!h) { fprintf(stderr, "dlopen(%s) failed: %s\n", lib, dlerror()); return 1; }
        fwd = (forward_argv_fn) dlsym(h, "tsi_forward_argv");
        if (!fwd) { fprintf(stderr, "dlsym tsi_forward_argv failed: %s\n", dlerror()); return 1; }
    }

    DecodeModel M = load_decode_model(path);
    const int L = L_arg > 0 ? L_arg : (N + gen + 2);
    const int VOC = M.n_vocab, KV = M.hidden_kv;
    fprintf(stderr, "dims: layers=%d hidden=%d n_head=%d/%d head_dim=%d vocab=%d L=%d prompt=%d gen=%d\n",
            M.n_layers, M.hidden, M.n_head, M.n_head_kv, M.head_dim, VOC, L, N, gen);

    // build the fixed-L decode graph once; its leaf tensors are updated in place each step
    ggml_init_params dp { (size_t) 256 << 20, nullptr, false };
    ggml_context * dc = ggml_init(dp);
    ggml_tensor * id   = ggml_new_tensor_1d(dc, GGML_TYPE_I32, 1);
    ggml_tensor * pos  = ggml_new_tensor_1d(dc, GGML_TYPE_I32, 1);
    ggml_tensor * mask = ggml_new_tensor_2d(dc, GGML_TYPE_F32, L + 1, 1);
    std::vector<ggml_tensor *> cK(M.n_layers), cV(M.n_layers);
    for (int il = 0; il < M.n_layers; il++) {
        cK[il] = ggml_new_tensor_3d(dc, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L); memset(cK[il]->data, 0, ggml_nbytes(cK[il]));
        cV[il] = ggml_new_tensor_3d(dc, GGML_TYPE_F32, M.head_dim, M.n_head_kv, L); memset(cV[il]->data, 0, ggml_nbytes(cV[il]));
    }
    std::vector<ggml_tensor *> knew, vnew;
    ggml_tensor * logits = build_decode(dc, M, id, pos, mask, cK, cV, knew, vnew);
    ggml_cgraph * gf = ggml_new_graph_custom(dc, 16384, false);
    ggml_build_forward_expand(gf, logits);
    for (int il = 0; il < M.n_layers; il++) { ggml_build_forward_expand(gf, knew[il]); ggml_build_forward_expand(gf, vnew[il]); }

    // outputs in the order the emitter uses: [logits, k_new0, v_new0, k_new1, v_new1, ...]
    std::vector<ggml_tensor *> outs; outs.push_back(logits);
    for (int il = 0; il < M.n_layers; il++) { outs.push_back(knew[il]); outs.push_back(vnew[il]); }
    auto leafs = discover_leafs(gf);
    fprintf(stderr, "graph: leafs=%zu outputs=%zu\n", leafs.size(), outs.size());

    // handle only for a clean finalize: the runtime is already up (llama_backend_init), and
    // ggml_backend_free below runs hal_lib_deinit so the static destructor doesn't abort.
    ggml_backend_t tsi_be = ggml_backend_tsavorite_init();

    // --emit: write this graph as the multi-output MLIR func (same leaf order as the run below)
    if (emit) {
        std::vector<const ggml_tensor *> co(outs.begin(), outs.end());
        std::string txt = build_func_text_baked_multi(gf, "forward", leafs, {}, co);
        std::ofstream f(emit); f << "module {\n" << txt << "}\n";
        fprintf(stderr, "emitted decode graph: L=%d leafs=%zu outputs=%zu -> %s\n", L, leafs.size(), outs.size(), emit);
        if (!lib) { ggml_free(dc); ggml_free(M.wc); llama_model_free(lmodel); ggml_backend_free(tsi_be); return 0; }
    }

    // device buffers + descriptors: argv = [desc(leaf0..), desc(out0..)]
    std::set<const ggml_tensor *> runtime = { id, pos, mask };
    for (int il = 0; il < M.n_layers; il++) { runtime.insert(cK[il]); runtime.insert(cV[il]); }

    const size_t NL = leafs.size();
    std::vector<void *> args(NL + outs.size());
    std::vector<void *> dev_leaf(NL), dev_out(outs.size());
    std::vector<const ggml_tensor *> leaf_t(NL);
    std::vector<char> leaf_is_rt(NL);
    for (size_t i = 0; i < NL; i++) {
        const ggml_tensor * t = leafs[i];
        leaf_t[i] = t;
        leaf_is_rt[i] = runtime.count(t) ? 1 : 0;
        dev_leaf[i] = tsi_alloc((int64_t) ggml_nbytes(t));
        if (!leaf_is_rt[i]) memcpy(dev_leaf[i], t->data, ggml_nbytes(t));   // weights: copy once
        args[i] = make_desc(t, dev_leaf[i]);
    }
    for (size_t j = 0; j < outs.size(); j++) {
        dev_out[j] = tsi_alloc((int64_t) ggml_nbytes(outs[j]));
        args[NL + j] = make_desc(outs[j], dev_out[j]);
    }

    // autoregressive decode loop
    std::vector<int32_t> seq = ids;   // full sequence (prompt ++ produced), for input + verify prefill
    std::vector<float> lbuf(VOC);
    int mism = 0, verified = 0;
    const int steps = N + gen;
    printf("[decode] prompt ids:"); for (int t : ids) printf(" %d", t); printf("\n");

    for (int step = 0; step < steps; step++) {
        const int32_t tok = seq[step];      // token fed this step (prompt, then produced)
        const int cur = step;               // its position == current cache length

        ((int32_t *) id->data)[0]  = tok;
        ((int32_t *) pos->data)[0] = cur;
        float * mm = (float *) mask->data;
        for (int j = 0; j < L; j++) mm[j] = (j < cur) ? 0.0f : -INFINITY;   // valid cache slots [0,cur)
        mm[L] = 0.0f;                                                        // the new token is always valid

        for (size_t i = 0; i < NL; i++)                                     // push runtime inputs to device
            if (leaf_is_rt[i]) memcpy(dev_leaf[i], leaf_t[i]->data, ggml_nbytes(leaf_t[i]));

        fwd(args.data());

        memcpy(lbuf.data(), dev_out[0], (size_t) VOC * sizeof(float));       // logits out
        int pred = argmax(lbuf.data(), VOC);
        for (int il = 0; il < M.n_layers; il++) {                           // append this step's K/V at slot cur
            memcpy((char *) cK[il]->data + (size_t) cur * KV * sizeof(float), dev_out[1 + 2 * il], KV * sizeof(float));
            memcpy((char *) cV[il]->data + (size_t) cur * KV * sizeof(float), dev_out[2 + 2 * il], KV * sizeof(float));
        }

        const char * kind = (step < N) ? "prompt" : "gen";
        if (verify) {   // CPU prefill of seq[0..step] last position -> reference argmax
            ggml_init_params pp { (size_t) 256 << 20, nullptr, false };
            ggml_context * pc = ggml_init(pp);
            const int m = step + 1;
            ggml_tensor * idt  = ggml_new_tensor_1d(pc, GGML_TYPE_I32, m); memcpy(idt->data, seq.data(), (size_t) m * sizeof(int32_t));
            ggml_tensor * pos2 = ggml_new_tensor_1d(pc, GGML_TYPE_I32, m); for (int i = 0; i < m; i++) ((int32_t *) pos2->data)[i] = i;
            ggml_tensor * pmask = ggml_new_tensor_2d(pc, GGML_TYPE_F32, m, m);
            { float * pm = (float *) pmask->data; for (int q = 0; q < m; q++) for (int j = 0; j < m; j++) pm[q * m + j] = (j <= q) ? 0.f : -INFINITY; }
            ggml_tensor * cp = ggml_get_rows(pc, M.embd, idt);
            for (int il = 0; il < M.n_layers; il++) cp = build_layer(pc, cp, M.lw[il], pos2, pmask, M.head_dim, M.n_head, M.n_head_kv, m);
            ggml_tensor * lp = ggml_mul_mat(pc, M.oww, ggml_mul(pc, ggml_rms_norm(pc, cp, REAL_RMS_EPS), M.onw));
            ggml_cgraph * gpf = ggml_new_graph_custom(pc, 16384, false);
            ggml_build_forward_expand(gpf, lp);
            ggml_graph_compute_with_ctx(pc, gpf, 4);
            const float * pl = (const float *) lp->data + (size_t) (m - 1) * VOC;
            int refm = argmax(pl, VOC);
            double num = 0, den = 0; for (int v = 0; v < VOC; v++) { double d = (double) lbuf[v] - pl[v]; num += d * d; den += (double) pl[v] * pl[v]; }
            bool ok = (pred == refm); if (!ok) mism++; verified++;
            fprintf(stderr, "step %2d %-6s tok=%-5d cur=%2d: compiled=%d prefill=%d rel_sq_err=%.3g -> %s\n",
                    step, kind, tok, cur, pred, refm, den > 0 ? num / den : num, ok ? "MATCH" : "DIFFER");
            ggml_free(pc);
        } else {
            fprintf(stderr, "step %2d %-6s tok=%-5d cur=%2d -> pred=%d\n", step, kind, tok, cur, pred);
        }

        if (step + 1 < (int) seq.size()) continue;   // still consuming the prompt: next input is known
        seq.push_back(pred);                          // generation: feed the prediction back
        if ((int) seq.size() - N >= gen) break;       // produced `gen` tokens
        if (cur + 1 >= L) { fprintf(stderr, "reached cache cap L=%d\n", L); break; }
    }

    printf("[decode] produced ids:"); for (int i = N; i < (int) seq.size(); i++) printf(" %d", seq[i]); printf("\n");
    std::string text;
    for (int i = N; i < (int) seq.size(); i++) {
        char buf[256];
        int l = llama_token_to_piece(vocab, seq[i], buf, sizeof buf, 0, true);
        if (l > 0) text.append(buf, l);
    }
    printf("[decode] generated text:%s\n", text.c_str());
    if (verify) fprintf(stderr, "=== compiled-decode vs prefill: %d/%d MATCH ===\n", verified - mism, verified);

    for (void * d : dev_leaf) tsi_dealloc(d);
    for (void * d : dev_out)  tsi_dealloc(d);
    ggml_free(dc); ggml_free(M.wc);
    llama_model_free(lmodel);
    ggml_backend_free(tsi_be);   // hal_lib_deinit / tsi_finalize - clean runtime teardown
    return mism == 0 ? 0 : 1;
}
