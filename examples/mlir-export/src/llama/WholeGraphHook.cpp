// Whole-graph interception hooked into llama_context. See WholeGraphHook.h.
#include "tsi/llama/WholeGraphHook.h"

#include "tsi/export/Exporter.h"                 // exportGraph, discoverLeafs, mlir_export_error
#include "tsi/graph/LiveGraphBuilder.h"       // build_cachefree_from_live (Approach B2)
#include "TestModel.h"                // MemRefDescriptor<N>, tsi_alloc (via HostShimCAPI.h)
#include "ggml-cpu.h"                 // ggml_graph_compute_with_ctx (reconstruction CPU reference)

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string wg_dir()  { const char * d = getenv("TSI_WG_DIR");  return d ? d : "."; }
int         wg_skip() { const char * s = getenv("TSI_WG_SKIP"); return s ? atoi(s) : 0; }

// build a heap MemRefDescriptor<N> over device pointer `p`, matching the exporter's
// mlir_tensor_type: rank = ggml_n_dims, MLIR shape = ne reversed (row-major strides).
template <int N>
void * make_desc_n(const ggml_tensor * t, void * p) {
    auto * d = new MemRefDescriptor<N>();
    d->base = p;
    d->data = p;
    d->offset = 0;
    for (int i = 0; i < N; i++) d->shape[i] = t->ne[N - 1 - i];   // reversed ne
    d->strides[N - 1] = 1;
    for (int i = N - 2; i >= 0; i--) d->strides[i] = d->strides[i + 1] * d->shape[i + 1];
    return d;
}

void * make_desc(const ggml_tensor * t, void * p) {
    switch (ggml_n_dims(t)) {
        case 1:  return make_desc_n<1>(t, p);
        case 2:  return make_desc_n<2>(t, p);
        case 3:  return make_desc_n<3>(t, p);
        default: return make_desc_n<4>(t, p);
    }
}

// TSI_WHOLEGRAPH=dump: list every node and its srcs (op, type, shape) to $TSI_WG_DIR/graph.txt.
void dump_graph(struct ggml_cgraph * g, const std::string & dir) {
    auto nm = [](const ggml_tensor * t) { return t->name[0] ? t->name : "(unnamed)"; };
    auto shp = [](const ggml_tensor * t) {
        char b[64];
        snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]",
                 (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        return std::string(b);
    };
    const int n = ggml_graph_n_nodes(g);
    std::ostringstream o;
    o << "# whole-graph dump: " << n << " nodes\n";
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(g, i);
        const char * op = nd->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(nd))
                                                  : ggml_op_name(nd->op);
        o << "[" << i << "] " << op << "  " << ggml_type_name(nd->type) << " " << shp(nd)
          << "  '" << nm(nd) << "'\n";
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            ggml_tensor * sc = nd->src[s];
            if (sc) o << "      src" << s << ": " << ggml_op_name(sc->op) << " "
                      << ggml_type_name(sc->type) << " " << shp(sc) << "  '" << nm(sc) << "'\n";
        }
    }
    { std::ofstream f(dir + "/graph.txt"); f << o.str(); }
    fprintf(stderr, "[tsi-wholegraph] dumped %d-node graph -> %s/graph.txt\n", n, dir.c_str());
}

// dlopen $TSI_WG_LIB once and resolve tsi_forward_argv (the generated void** unpacking shim that
// forwards to the N-arg _mlir_ciface_forward). Using the shim avoids a libffi dependency: the arg
// count is baked into the shim at compile time by compile_graph_fpga.py.
typedef void (*forward_argv_fn)(void **);
forward_argv_fn load_forward() {
    static forward_argv_fn fwd = nullptr;
    static bool tried = false;
    if (tried) return fwd;
    tried = true;
    std::string lib = getenv("TSI_WG_LIB") ? getenv("TSI_WG_LIB")
                                           : wg_dir() + "/out_fpga/host/host.so";
    void * h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "[tsi-wholegraph] dlopen(%s) failed: %s\n", lib.c_str(), dlerror()); return nullptr; }
    fwd = (forward_argv_fn) dlsym(h, "tsi_forward_argv");
    if (!fwd) fprintf(stderr, "[tsi-wholegraph] dlsym tsi_forward_argv failed: %s "
                              "(rebuild host.so with the updated compile_graph_fpga.py)\n", dlerror());
    return fwd;
}

// Read the target LIVE node count from $TSI_WG_DIR/forward.manifest ("# live_nodes=<N> ...").
int manifest_nodes() {
    std::ifstream mf(wg_dir() + "/forward.manifest");
    std::string tok, kv;
    if (mf >> tok && mf >> kv) {
        auto p = kv.find('=');
        if (p != std::string::npos) return atoi(kv.c_str() + p + 1);
    }
    return -1;
}

} // namespace

// Scheduler eval-callback. Fires per node after it computes (ask=false), while its inputs are still
// valid. We snapshot each weight leaf here because the reconstruction can't read it later: the
// scheduler's CPU# copies sit in recycled scratch and q/k/v have no persistent leaf.
std::map<std::string, std::vector<float>> g_wcap;

// The token ids, snapshotted for the same reason as the weights. build_cachefree_from_live rebuilds
// positions and the causal mask arithmetically, so those need no snapshot, but it copies the ids out
// of the live buffer - which is recycled scratch by the time maybe_run() executes. Without this the
// reconstruction reads freed memory as token ids and ggml_get_rows aborts on an out-of-range index.
std::vector<int32_t> g_ids_cap;

// The real positions, captured for validation only (see the ROPE branch below).
std::vector<int32_t> g_pos_cap;

extern "C" bool tsi_wholegraph_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (ask) return true;
    if (!t) return true;

    // grab each weight leaf once, when its first consumer op runs (keyed by core name)
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        ggml_tensor * sc = t->src[s];
        if (!sc || sc->op != GGML_OP_NONE || sc->type != GGML_TYPE_F32 || !sc->data) continue;
        std::string cn = wg_core_name(sc->name);
        if (cn.size() >= 7 && cn.compare(cn.size() - 7, 7, ".weight") == 0 && !g_wcap.count(cn)) {
            int64_t n = ggml_nelements(sc);
            std::vector<float> & v = g_wcap[cn];
            v.resize((size_t) n);
            memcpy(v.data(), sc->data, (size_t) n * sizeof(float));
        }
    }

    // Token ids, identified exactly as build_cachefree_from_live identifies them: the i32 src1 of the
    // GET_ROWS that reads the embedding table. Structural rather than by name, because llama names
    // this input differently across versions.
    if (g_ids_cap.empty() && t->op == GGML_OP_GET_ROWS && t->src[0] && t->src[1] &&
        wg_core_name(t->src[0]->name) == "token_embd.weight" &&
        t->src[1]->type == GGML_TYPE_I32 && t->src[1]->data) {
        const int64_t n = ggml_nelements(t->src[1]);
        g_ids_cap.resize((size_t) n);
        memcpy(g_ids_cap.data(), t->src[1]->data, (size_t) n * sizeof(int32_t));
    }

    // Positions, from the first ROPE node - same place build_cachefree_from_live reads them. Needed
    // only so that reconstruction can *check* them: it rebuilds positions as 0..n-1, and must refuse
    // a graph whose real positions differ (a decode step, or a continued prefill) instead of
    // silently emitting a graph that computes something else.
    if (g_pos_cap.empty() && t->op == GGML_OP_ROPE && t->src[1] &&
        t->src[1]->type == GGML_TYPE_I32 && t->src[1]->data) {
        const int64_t n = ggml_nelements(t->src[1]);
        g_pos_cap.resize((size_t) n);
        memcpy(g_pos_cap.data(), t->src[1]->data, (size_t) n * sizeof(int32_t));
    }
    return true;
}

void tsi_wholegraph_maybe_capture(struct ggml_cgraph * cgraph) {
    const char * mode = getenv("TSI_WHOLEGRAPH");
    if (!mode) return;
    const std::string m = mode;
    const bool is_dump    = (m == "dump");
    const bool is_capture = (m == "capture");
    if (!is_dump && !is_capture) return;

    static int seen = 0;
    static bool done = false;
    if (done) return;
    if (seen++ < wg_skip()) return;          // skip warmup graphs
    done = true;                             // one attempt regardless of outcome

    const std::string dir = wg_dir();

    if (is_dump) {                           // diagnostic only: list ops/shapes, no emit
        dump_graph(cgraph, dir);
        return;
    }

    // The live graph's in-place KV cache can't be a pure MLIR func, so export the equivalent
    // cache-free prefill instead (build_cachefree_from_live). Throws on an unexpected shape.
    case_result r;
    try {
        r = build_cachefree_from_live(cgraph);
    } catch (const std::exception & e) {
        fprintf(stderr,
                "[tsi-wholegraph] capture SKIPPED: %s. (Run TSI_WHOLEGRAPH=dump to inspect.) "
                "Continuing per-op.\n", e.what());
        return;
    }
    std::string module = r.func_text;   // exportGraph already returns a complete module

    { std::ofstream f(dir + "/forward.mlir"); f << module; }
    {
        std::ofstream mf(dir + "/forward.manifest");
        // live_nodes identifies the prefill call at run time; args = the reconstructed runtime args.
        mf << "# live_nodes=" << ggml_graph_n_nodes(cgraph)
           << " recon_nodes=" << ggml_graph_n_nodes(r.gf)
           << " args=" << r.runtime_args.size() << "\n";
        for (const ggml_tensor * t : r.runtime_args) {
            mf << wg_core_name(t->name) << " ndims=" << ggml_n_dims(t)
               << " nbytes=" << ggml_nbytes(t) << "\n";
        }
    }
    fprintf(stderr,
            "[tsi-wholegraph] captured (cache-free reconstruction): %d nodes, %zu args -> %s/forward.mlir\n",
            ggml_graph_n_nodes(r.gf), r.runtime_args.size(), dir.c_str());
    ggml_free(r.ctx);
}

// Called AFTER the scheduler ran the per-op path (so the TSI runtime is initialized and the live
// output tensor holds the per-op logits = reference). Reconstructs the cache-free graph (same as
// capture), binds its args to device buffers, calls the compiled forward, and compares the
// last-token logits/argmax against the per-op reference. verify: report only. run: also overwrite
// the live output so llama samples the compiled token.
bool tsi_wholegraph_maybe_run(struct ggml_cgraph * live) {
    const char * mode = getenv("TSI_WHOLEGRAPH");
    if (!mode) return false;
    const std::string m = mode;
    const bool is_run    = (m == "run");
    const bool is_verify = (m == "verify");
    if (!is_run && !is_verify) return false;

    static bool done = false;
    if (done) return false;
    const int want = manifest_nodes();
    static int seen = 0;
    if (want >= 0) { if (ggml_graph_n_nodes(live) != want) return false; }
    else           { if (seen++ < wg_skip()) return false; }
    done = true;

    case_result r;
    try {
        r = build_cachefree_from_live(live);
    } catch (const std::exception & e) {
        fprintf(stderr, "[tsi-wholegraph] %s SKIPPED: %s\n", m.c_str(), e.what());
        return false;
    }

    forward_argv_fn fwd = load_forward();
    if (!fwd) { ggml_free(r.ctx); return false; }

    ggml_tensor *  rout  = ggml_graph_node(r.gf, -1);   // reconstructed logits [n_vocab, n_tokens]
    const int64_t  nvoc  = rout->ne[0];
    const int64_t  ntok  = rout->ne[1];
    const int64_t  n_out = ggml_nelements(rout);

    // Reconstruction's own CPU result (ggml on r.gf). Lets a mismatch be pinned to either the
    // reconstruction math (recon-CPU vs per-op) or the TSI compile (compiled vs recon-CPU).
    std::vector<float> reconcpu((size_t) n_out);
    ggml_graph_compute_with_ctx(r.ctx, r.gf, 4);
    memcpy(reconcpu.data(), rout->data, (size_t) n_out * sizeof(float));
    const float * rcpu_last = reconcpu.data() + (size_t) (ntok - 1) * nvoc;

    fprintf(stderr, "[tsi-wholegraph] recon per-column argmax:");
    for (int64_t t = 0; t < ntok; t++) {
        const float * col = reconcpu.data() + (size_t) t * nvoc;
        int64_t am = 0; for (int64_t v = 1; v < nvoc; v++) if (col[v] > col[am]) am = v;
        fprintf(stderr, " %lld", (long long) am);
    }
    fprintf(stderr, "\n");

    // Bring the TSI host runtime up before the first tsi_alloc. On a TSI build the ggml-tsavorite
    // backend has already done this during llama_backend_init, but a plain host/FFM build has no
    // such backend, and calling tsi_alloc on an uninitialized runtime segfaults. Initialize once per
    // process; tsi_finalize runs after the compiled call below.
    static bool rt_up = false;
    if (!rt_up) {
        tsi_initialize(1);
        rt_up = true;
        fprintf(stderr, "[tsi-wholegraph] TSI host runtime initialized (1 TXE)\n");
    }

    // copy every arg into a device buffer the Xtensa blob can read (host pointers won't do)
    const size_t N = r.runtime_args.size();
    std::vector<void *> argv(N + 1);
    std::vector<void *> devbufs;
    for (size_t i = 0; i < N; i++) {
        const ggml_tensor * t = r.runtime_args[i];
        size_t nb  = ggml_nbytes(t);
        void * dev = tsi_alloc((int64_t) nb);
        if (!dev) {
            // Out of simulated DRAM: report it rather than segfaulting in the memcpy below.
            fprintf(stderr, "[tsi-wholegraph] tsi_alloc failed for arg %zu (%zu bytes). Raise "
                            "USER_DRAM_SIZE (MiB) and retry.\n", i, nb);
            for (void * d : devbufs) tsi_dealloc(d);
            ggml_free(r.ctx);
            tsi_finalize();   // else the process will not exit
            rt_up = false;
            return false;
        }
        memcpy(dev, t->data, nb);
        devbufs.push_back(dev);
        argv[i] = make_desc(t, dev);
    }
    void * dev_out = tsi_alloc(n_out * (int64_t) sizeof(float));
    if (!dev_out) {
        fprintf(stderr, "[tsi-wholegraph] tsi_alloc failed for the output buffer (%lld bytes). "
                        "Raise USER_DRAM_SIZE (MiB) and retry.\n", (long long) (n_out * 4));
        for (void * d : devbufs) tsi_dealloc(d);
        ggml_free(r.ctx);
        tsi_finalize();   // else the process will not exit
        rt_up = false;
        return false;
    }
    devbufs.push_back(dev_out);
    argv[N] = make_desc(rout, dev_out);

    fprintf(stderr, "[tsi-wholegraph] running compiled forward: %zu args, logits [%lld x %lld]\n",
            N, (long long) nvoc, (long long) ntok);
    fwd(argv.data());

    // compiled next-token logits = last column of [n_vocab, n_tokens]
    std::vector<float> compiled((size_t) n_out);
    memcpy(compiled.data(), dev_out, (size_t) n_out * sizeof(float));
    const float * clast = compiled.data() + (size_t) (ntok - 1) * nvoc;

    // reference = the per-op path's output tensor (last node of the live graph), last-token logits
    ggml_tensor * live_out = ggml_graph_node(live, -1);
    const float * ref = (const float *) live_out->data;
    const int64_t ref_n = ggml_nelements(live_out);   // = n_vocab (llama emits only the last token)
    const int64_t n = ref_n < nvoc ? ref_n : nvoc;

    // 3-way diagnostic over the next-token logits.
    auto compare = [&](const char * label, const float * a, const float * b) {
        double num = 0.0, den = 0.0, maxabs = 0.0;
        int64_t amax_a = 0, amax_b = 0;
        for (int64_t v = 0; v < n; v++) {
            double d = (double) a[v] - (double) b[v];
            num += d * d; den += (double) b[v] * (double) b[v];
            if (d < 0) d = -d;
            if (d > maxabs) maxabs = d;
            if (a[v] > a[amax_a]) amax_a = v;
            if (b[v] > b[amax_b]) amax_b = v;
        }
        double rel = den > 0.0 ? num / den : num;
        fprintf(stderr, "[tsi-wholegraph] VERIFY %-22s rel_sq_err=%-10g max_abs=%-10g argmax %lld vs %lld  -> %s\n",
                label, rel, maxabs, (long long) amax_a, (long long) amax_b,
                (amax_a == amax_b) ? "MATCH" : "DIFFER");
    };
    compare("recon-CPU vs per-op:",  rcpu_last, ref);        // is the reconstruction math right?
    compare("compiled vs recon-CPU:", clast,    rcpu_last);  // is the TSI compile right?
    compare("compiled vs per-op:",    clast,    ref);        // end-to-end

    if (is_run) {   // feed llama the compiled next-token logits so it samples the compiled token
        memcpy(live_out->data, clast, (size_t) n * sizeof(float));
    }

    for (void * d : devbufs) tsi_dealloc(d);
    ggml_free(r.ctx);

    // Tear the runtime down explicitly. Without this the process does not exit: the runtime keeps
    // state alive past main() and the wait never completes, so llama-cli appears to hang long after
    // it has printed its final timings. The verify/run hook fires once, so finalizing here is safe.
    tsi_finalize();
    rt_up = false;

    return false;   // per-op already ran; nothing for the caller to skip
}
