// The driver: classify each intercepted graph, then export -> compile -> run it.
//
// Orchestration only. Rebuilding the graph is LiveGraphBuilder's job, emitting MLIR is the
// exporter's, compiling is Artifact's, and moving bytes to the device is DeviceArgs'.
#include "tsi/driver/ExportDriver.h"

#include "Artifact.h"
#include "Config.h"
#include "DeviceArgs.h"
#include "Runtime.h"

#include "tsi/graph/LiveGraphBuilder.h"   // build_cachefree_from_live, case_result
#include "ggml-cpu.h"                     // ggml_graph_compute_with_ctx (CPU reference)

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace tsi::driver;

// ---------------------------------------------------------------------------------------------
// snapshots taken during compute, while the live buffers are still valid
// ---------------------------------------------------------------------------------------------

// Each weight leaf, grabbed once when its first consumer runs. The reconstruction cannot read these
// later: the scheduler's CPU# copies sit in recycled scratch, and q/k/v have no persistent leaf.
std::map<std::string, std::vector<float>> g_wcap;

// Token ids, for the same reason. build_cachefree_from_live rebuilds positions and the causal mask
// arithmetically, but it copies the ids out of the live buffer, which is recycled by then. Without
// this the reconstruction reads freed memory as token ids and ggml_get_rows aborts.
std::vector<int32_t> g_ids_cap;

// The real positions. Used to tell prefill from decode, and to refuse a graph whose positions are
// not 0..n-1 rather than silently exporting something that computes a different thing.
std::vector<int32_t> g_pos_cap;

namespace {

// ---------------------------------------------------------------------------------------------
// phase
// ---------------------------------------------------------------------------------------------

enum class Phase { Skip, Prefill, Decode };

// Prefill starts at position 0; decode continues from wherever llama is. Read from the snapshot
// rather than the graph shape, because a 1-token graph is prefill when the prompt is one token.
Phase classify() {
    if (g_pos_cap.empty()) {
        return Phase::Skip;   // no ROPE seen, so not a transformer forward we understand
    }
    return g_pos_cap[0] == 0 ? Phase::Prefill : Phase::Decode;
}

const char * phaseName(Phase p) {
    switch (p) {
        case Phase::Prefill: return "prefill";
        case Phase::Decode:  return "decode";
        default:             return "skip";
    }
}

// ---------------------------------------------------------------------------------------------
// diagnostics
// ---------------------------------------------------------------------------------------------

void dumpGraph(ggml_cgraph * g, const std::string & path) {
    auto shp = [](const ggml_tensor * t) {
        char b[64];
        snprintf(b, sizeof b, "[%lld,%lld,%lld,%lld]", (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3]);
        return std::string(b);
    };
    std::ofstream o(path);
    const int     n = ggml_graph_n_nodes(g);
    o << "# graph dump: " << n << " nodes\n";
    for (int i = 0; i < n; i++) {
        ggml_tensor * nd = ggml_graph_node(g, i);
        const char *  op = nd->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(nd))
                                                   : ggml_op_name(nd->op);
        o << "[" << i << "] " << op << "  " << ggml_type_name(nd->type) << " " << shp(nd) << "  '"
          << (nd->name[0] ? nd->name : "(unnamed)") << "'\n";
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (ggml_tensor * sc = nd->src[s]) {
                o << "      src" << s << ": " << ggml_op_name(sc->op) << " "
                  << ggml_type_name(sc->type) << " " << shp(sc) << "  '"
                  << (sc->name[0] ? sc->name : "(unnamed)") << "'\n";
            }
        }
    }
    fprintf(stderr, "[tsi-mlir] dumped %d-node graph -> %s\n", n, path.c_str());
}

// Relative squared error and argmax agreement over one token's logits.
void compare(const char * label, const float * a, const float * b, int64_t n) {
    double  num = 0.0, den = 0.0, maxabs = 0.0;
    int64_t amax_a = 0, amax_b = 0;
    for (int64_t v = 0; v < n; v++) {
        double d = (double) a[v] - (double) b[v];
        num += d * d;
        den += (double) b[v] * (double) b[v];
        if (d < 0) {
            d = -d;
        }
        if (d > maxabs) {
            maxabs = d;
        }
        if (a[v] > a[amax_a]) {
            amax_a = v;
        }
        if (b[v] > b[amax_b]) {
            amax_b = v;
        }
    }
    fprintf(stderr, "[tsi-mlir] %-24s rel_sq_err=%-11g max_abs=%-11g argmax %lld vs %lld -> %s\n",
            label, den > 0.0 ? num / den : num, maxabs, (long long) amax_a, (long long) amax_b,
            amax_a == amax_b ? "MATCH" : "DIFFER");
}

// ---------------------------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------------------------

// Rebuild, compile, run. Returns the compiled next-token logits, or empty on any failure, in which
// case the caller leaves llama's own result in place.
std::vector<float> runPrefill(ggml_cgraph * live, const Config & cfg, int64_t & nvoc_out) {
    case_result r;
    try {
        r = build_cachefree_from_live(live);
    } catch (const std::exception & e) {
        fprintf(stderr, "[tsi-mlir] prefill SKIPPED: %s\n", e.what());
        return {};
    }

    forward_argv_fn fwd = buildForward(r.func_text, "prefill", cfg);
    if (!fwd) {
        ggml_free(r.ctx);
        return {};
    }

    ggml_tensor * rout  = ggml_graph_node(r.gf, -1);   // logits [n_vocab, n_tokens]
    const int64_t nvoc  = rout->ne[0];
    const int64_t ntok  = rout->ne[1];
    const int64_t n_out = ggml_nelements(rout);
    nvoc_out            = nvoc;

    // The reconstruction's own CPU result. Only under TSI_MLIR_CPU_REF: it is a full extra forward
    // pass, and it exists to split a reconstruction bug from a compilation bug, which the 2-way diff
    // rarely fails to locate on its own.
    std::vector<float> reconcpu;
    if (cfg.cpu_ref) {
        reconcpu.resize((size_t) n_out);
        ggml_graph_compute_with_ctx(r.ctx, r.gf, 4);
        memcpy(reconcpu.data(), rout->data, (size_t) n_out * sizeof(float));
    }

    runtimeUp();

    DeviceArgs args;
    bool       ok = true;
    for (const ggml_tensor * t : r.runtime_args) {
        ok = ok && args.addInput(t);
    }
    ok = ok && args.addOutput(rout);
    if (!ok) {
        ggml_free(r.ctx);
        return {};
    }

    fprintf(stderr, "[tsi-mlir] running compiled prefill: %zu args, logits [%lld x %lld]\n",
            r.runtime_args.size(), (long long) nvoc, (long long) ntok);
    fwd(args.argv());

    std::vector<float> compiled((size_t) n_out);
    memcpy(compiled.data(), args.buffer(r.runtime_args.size()), (size_t) n_out * sizeof(float));

    // llama emits only the last token's logits, so that is what the diffs compare.
    const float * clast = compiled.data() + (size_t) (ntok - 1) * nvoc;
    if (cfg.cpu_ref) {
        const float * rlast = reconcpu.data() + (size_t) (ntok - 1) * nvoc;
        compare("recon-CPU vs compiled:", rlast, clast, nvoc);
    }

    std::vector<float> last(clast, clast + nvoc);
    ggml_free(r.ctx);
    return last;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------------------------

extern "C" bool tsi_mlir_export_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    if (ask || !t) {
        return true;
    }

    for (int s = 0; s < GGML_MAX_SRC; s++) {
        ggml_tensor * sc = t->src[s];
        if (!sc || sc->op != GGML_OP_NONE || sc->type != GGML_TYPE_F32 || !sc->data) {
            continue;
        }
        std::string cn = wg_core_name(sc->name);
        if (cn.size() >= 7 && cn.compare(cn.size() - 7, 7, ".weight") == 0 && !g_wcap.count(cn)) {
            std::vector<float> & v = g_wcap[cn];
            v.resize((size_t) ggml_nelements(sc));
            memcpy(v.data(), sc->data, v.size() * sizeof(float));
        }
    }

    // Token ids: the i32 src1 of the GET_ROWS that reads the embedding table. Structural rather than
    // by name, because llama names this input differently across versions.
    if (g_ids_cap.empty() && t->op == GGML_OP_GET_ROWS && t->src[0] && t->src[1] &&
        wg_core_name(t->src[0]->name) == "token_embd.weight" && t->src[1]->type == GGML_TYPE_I32 &&
        t->src[1]->data) {
        g_ids_cap.resize((size_t) ggml_nelements(t->src[1]));
        memcpy(g_ids_cap.data(), t->src[1]->data, g_ids_cap.size() * sizeof(int32_t));
    }

    // Positions, from the first ROPE node.
    if (g_pos_cap.empty() && t->op == GGML_OP_ROPE && t->src[1] &&
        t->src[1]->type == GGML_TYPE_I32 && t->src[1]->data) {
        g_pos_cap.resize((size_t) ggml_nelements(t->src[1]));
        memcpy(g_pos_cap.data(), t->src[1]->data, g_pos_cap.size() * sizeof(int32_t));
    }
    return true;
}

void tsi_mlir_export_before_compute(struct ggml_cgraph * cgraph) {
    const Config & cfg = Config::get();
    if (!cfg.enabled) {
        return;
    }
    // Each graph gets fresh snapshots: ids and positions differ per step, and a stale position would
    // misclassify the phase.
    g_ids_cap.clear();
    g_pos_cap.clear();
    (void) cgraph;
}

bool tsi_mlir_export_after_compute(struct ggml_cgraph * live) {
    const Config & cfg = Config::get();
    if (!cfg.enabled) {
        return false;
    }

    static int seen = 0;
    if (seen++ < cfg.skip) {
        fprintf(stderr, "[tsi-mlir] skipping graph %d (%d nodes), llama's warmup\n", seen - 1,
                ggml_graph_n_nodes(live));
        return false;
    }

    const Phase phase = classify();
    if (cfg.dump) {
        dumpGraph(live, cfg.dir + "/graph-" + phaseName(phase) + ".txt");
    }

    // One shot per phase. Prefill happens once; decode repeats per token, and running the compiled
    // decode on every one of them is Step 5b's business, once the graph exists.
    static bool did_prefill = false;
    static bool did_decode  = false;

    if (phase == Phase::Prefill) {
        if (did_prefill) {
            return false;
        }
        did_prefill = true;

        int64_t            nvoc     = 0;
        std::vector<float> compiled = runPrefill(live, cfg, nvoc);
        if (compiled.empty()) {
            return false;
        }

        ggml_tensor * live_out = ggml_graph_node(live, -1);
        const int64_t n        = ggml_nelements(live_out) < nvoc ? ggml_nelements(live_out) : nvoc;
        if (cfg.verify) {
            compare("compiled vs llama:", compiled.data(), (const float *) live_out->data, n);
        }
        // The compiled logits are the result. llama samples from these and continues.
        memcpy(live_out->data, compiled.data(), (size_t) n * sizeof(float));
        return false;
    }

    if (phase == Phase::Decode && !did_decode) {
        did_decode = true;
        fprintf(stderr, "[tsi-mlir] decode graph seen (%d nodes, pos %d). The compiled decode path "
                        "is not wired up yet; llama's own result stands.\n",
                ggml_graph_n_nodes(live), g_pos_cap.empty() ? -1 : g_pos_cap[0]);
    }
    return false;
}
