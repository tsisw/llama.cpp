// The driver: classify each intercepted graph, then export -> compile -> run it.
//
// Orchestration only. Rebuilding the graph is LiveGraphBuilder's job, emitting MLIR is the
// exporter's, compiling is Artifact's, and moving bytes to the device is DeviceArgs'.
#include "tsi/driver/ExportDriver.h"

#include "Artifact.h"
#include "Config.h"
#include "DeviceArgs.h"
#include "Runtime.h"

#include "tsi/graph/LiveCache.h"          // live_cache_probe / _extract (llama's KV cache)
#include "tsi/graph/LiveDecodeBuilder.h"   // build_decode_from_live, decode_case
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

// The cells llama's own allocator picked for this batch, read off the SET_ROWS that writes the cache.
//
// These are NOT the positions. llama's slot allocator finds free cells, and after a removal, a defrag
// or in a multi-sequence context they diverge from pos. `k_idxs` is a graph input filled from slot_info
// before compute, so it is readable here, and using it is what keeps our writes landing where llama
// expects the tokens to be. One entry per token: n_tokens for prefill, 1 for decode.
//
// The SET_ROWS node carries the cache name itself and takes [values, indices, cache], so the indices
// are src[1]. Read off a dumped graph rather than assumed from the ggml_set_rows signature.
bool liveCacheCells(ggml_cgraph * live, std::vector<int> & out) {
    out.clear();
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op != GGML_OP_SET_ROWS || !nd->src[1]) {
            continue;
        }
        if (std::string(nd->name).rfind("cache_k_l", 0) != 0) {
            continue;
        }
        ggml_tensor * idx = nd->src[1];
        if (idx->type != GGML_TYPE_I64 || !idx->data) {
            continue;
        }
        const int64_t n = ggml_nelements(idx);
        for (int64_t t = 0; t < n; t++) {
            out.push_back((int) ((const int64_t *) idx->data)[t]);
        }
        return !out.empty();
    }
    return false;
}

// Write a graph's per-layer K/V results into llama's cache, one cell per token.
//
// Shared by both phases, because they differ only in the token count: prefill returns n_tokens cells
// per layer and decode returns one. Results are ordered [logits, k_new_0..N-1, v_new_0..N-1], and a
// layer's result holds token t's cell contiguously at t * per_cell, which is llama's own cell layout.
// The graph computes f32 and llama's cache is f16, so this is also where the narrowing happens - the
// same rounding llama's SET_ROWS would have applied.
bool writeCacheCells(ggml_cgraph * live, DeviceArgs & args, int n_layers,
                     const std::vector<int> & cells) {
    ggml_tensor * k0 = live_cache_leaf(live, "k", 0);
    if (!k0 || n_layers <= 0 || cells.empty()) {
        return false;
    }
    const ggml_type ctype      = k0->type;
    const int64_t   per_cell   = k0->ne[0];
    const int64_t   capacity   = k0->ne[1];
    const size_t    cell_bytes = (size_t) per_cell * ggml_type_size(ctype);

    if (ctype != GGML_TYPE_F16 && ctype != GGML_TYPE_F32) {
        fprintf(stderr, "[tsi-mlir] cannot write a %s cache\n", ggml_type_name(ctype));
        return false;
    }

    for (int kind = 0; kind < 2; kind++) {
        for (int il = 0; il < n_layers; il++) {
            ggml_tensor * lt = live_cache_leaf(live, kind == 0 ? "k" : "v", il);
            if (!lt || !lt->data || lt->type != ctype || lt->ne[0] != per_cell) {
                fprintf(stderr, "[tsi-mlir] cache_%s_l%d is not the geometry layer 0 declared\n",
                        kind == 0 ? "k" : "v", il);
                return false;
            }
            const float * src = (const float *) args.output(1 + (size_t) kind * n_layers + il);
            for (size_t t = 0; t < cells.size(); t++) {
                if (cells[t] < 0 || cells[t] >= capacity) {
                    fprintf(stderr, "[tsi-mlir] cell %d is outside the %lld-cell buffer\n", cells[t],
                            (long long) capacity);
                    return false;
                }
                const float * s   = src + (size_t) t * per_cell;
                char *        dst = (char *) lt->data + (size_t) cells[t] * cell_bytes;
                if (ctype == GGML_TYPE_F16) {
                    ggml_fp32_to_fp16_row(s, (ggml_fp16_t *) dst, per_cell);
                } else {
                    memcpy(dst, s, cell_bytes);
                }
            }
        }
    }
    return true;
}

// Rebuild, compile, run. Returns the compiled next-token logits, or empty on any failure, in which
// case the caller leaves llama's own result in place.
std::vector<float> runPrefill(ggml_cgraph * live, const Config & cfg, int64_t & nvoc_out) {
    case_result r;
    try {
        r = build_cachefree_from_live(live, cfg.weight_args);
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
    // [logits, k_new_0..N-1, v_new_0..N-1]: the prompt's K/V come back so the compiled graph, not
    // llama, is what authors cells 0..n_tokens-1 of the cache.
    for (const ggml_tensor * t : r.outputs) {
        ok = ok && args.addOutput(t);
    }
    if (!ok) {
        ggml_free(r.ctx);
        return {};
    }

    fprintf(stderr, "[tsi-mlir] running compiled prefill: %zu args, %zu results, logits [%lld x %lld]\n",
            r.runtime_args.size(), r.outputs.size(), (long long) nvoc, (long long) ntok);
    fwd(args.argv());

    // Author the prompt's cells from the compiled result, at the cells llama assigned. llama's own
    // SET_ROWS also ran - its prefill pass is not skipped, because the weight snapshot depends on it -
    // so this overwrites those cells and the whole cache ends up with a single author.
    //
    // Not under TSI_MLIR_VERIFY: there llama has to stay an independent reference, and overwriting the
    // prompt's cells would leave llama's own decode reading values we produced. Same reason decode's
    // write is gated. The cache path is exercised by running without verify and checking the generated
    // text still matches the verify run's.
    std::vector<int> cells;
    if (!r.k_new.empty() && !cfg.verify) {
        if (!liveCacheCells(live, cells) || cells.size() != (size_t) ntok) {
            fprintf(stderr, "[tsi-mlir] prefill: got %zu cache cells for %lld tokens; leaving llama's "
                            "own cache values in place\n", cells.size(), (long long) ntok);
        } else if (writeCacheCells(live, args, (int) r.k_new.size(), cells)) {
            fprintf(stderr, "[tsi-mlir] prefill wrote %zu cells x %zu layers into llama's cache\n",
                    cells.size(), r.k_new.size());
        }
    }

    std::vector<float> compiled((size_t) n_out);
    memcpy(compiled.data(), args.output(0), (size_t) n_out * sizeof(float));

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

// ---------------------------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------------------------

// Read llama's KV cache and report what was found, without building anything yet.
//
// This runs on the real decode graph, so the geometry the next step will rely on is measured here
// rather than assumed. It also checks the one thing that could silently be wrong: that the cells
// llama has written are non-zero and the cells beyond its position are still zero. If the layout
// assumption in LiveCache.h were off, the boundary would land somewhere else and this would say so.
void reportDecodeCache(ggml_cgraph * live) {
    const int pos = g_pos_cap.empty() ? -1 : g_pos_cap[0];
    fprintf(stderr, "[tsi-mlir] decode graph: %d nodes, pos %d\n", ggml_graph_n_nodes(live), pos);

    // Geometry straight off the VIEW a decode step reads: [head_dim, n_head_kv, n_kv].
    int head_dim = 0, n_head_kv = 0;
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_VIEW && nd->src[0] && ggml_n_dims(nd) == 3 &&
            wg_core_name(nd->src[0]->name).rfind("cache_k_l", 0) == 0) {
            head_dim  = (int) nd->ne[0];
            n_head_kv = (int) nd->ne[1];
            break;
        }
    }
    if (head_dim == 0) {
        fprintf(stderr, "[tsi-mlir] no cache_k_l* view found; this is not a KV-cache decode graph\n");
        return;
    }

    const LiveCacheInfo k = live_cache_probe(live, "k", head_dim, n_head_kv);
    const LiveCacheInfo v = live_cache_probe(live, "v", head_dim, n_head_kv);
    if (!k.found || !v.found) {
        fprintf(stderr, "[tsi-mlir] cache_k/cache_v not both found\n");
        return;
    }
    fprintf(stderr, "[tsi-mlir] cache: %d layers, head_dim %d, n_head_kv %d, capacity %d cells, "
                    "live window n_kv %d\n",
            k.n_layers, k.head_dim, k.n_head_kv, k.n_cells, k.n_kv);

    // Extract layer 0's window and look at where the written cells stop.
    ggml_init_params ip { (size_t) 64 << 20, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);
    const int        L   = k.n_kv > 0 ? k.n_kv : k.n_cells;
    ggml_tensor *    ck  = live_cache_extract(ctx, live, "k", 0, k, L);
    if (!ck) {
        ggml_free(ctx);
        return;
    }

    const int64_t per_cell = (int64_t) k.head_dim * k.n_head_kv;
    const float * d        = (const float *) ck->data;
    int           last_nz  = -1;
    for (int c = 0; c < L; c++) {
        for (int64_t e = 0; e < per_cell; e++) {
            if (d[(size_t) c * per_cell + e] != 0.0f) {
                last_nz = c;
                break;
            }
        }
    }
    // Called before compute, so llama has not yet written this step's cell: cells 0..pos-1 hold the
    // prompt and everything from pos on is still zero. That is exactly the decode input we want, and
    // it is why the snapshot cannot wait until after compute - by then cell pos is written too and the
    // graph would be reading its own answer.
    const bool as_expected = last_nz == pos - 1;
    fprintf(stderr, "[tsi-mlir] cache_k_l0: last written cell %d, pos %d -> %s\n", last_nz, pos,
            as_expected ? "layout CONFIRMED" : "UNEXPECTED (layout assumption is wrong)");
    ggml_free(ctx);
}

// Compiled decode logits, produced in before_compute and consumed in after_compute. They cannot be
// written straight into the live output: llama has not computed it yet, and its own pass would
// overwrite them.
std::vector<float> g_decode_logits;
int64_t            g_decode_nvoc = 0;

// ---------------------------------------------------------------------------------------------
// llama's KV cache, aliased in place
// ---------------------------------------------------------------------------------------------

// Collect llama's own per-layer cache buffers, in the order the exporter declared them:
// cache_k_0..N-1 then cache_v_0..N-1.
//
// Nothing is allocated and nothing is copied. These are llama's tensors, which live in TSI DRAM
// because of the buffer type in KvBuffer.cpp, and the compiled graph reads them where they are. That
// is the whole point of the design: one cache, llama writes it, we read it, so every cache operation
// llama performs keeps working because nothing else mutates it.
bool collectCachePtrs(ggml_cgraph * live, const decode_case & r, std::vector<void *> & out) {
    out.clear();
    const size_t want = (size_t) r.cache_shape[1] * r.cache_shape[2] * r.cache_shape[3] *
                        ggml_type_size(r.cache_type);

    for (int kind = 0; kind < 2; kind++) {
        const char * k = kind == 0 ? "k" : "v";
        for (int il = 0; il < r.n_layers; il++) {
            ggml_tensor * t = live_cache_leaf(live, k, il);
            if (!t || !t->data) {
                fprintf(stderr, "[tsi-mlir] cache_%s_l%d missing; cannot alias llama's cache\n", k, il);
                return false;
            }
            // A size or type disagreement means the memref would address the wrong bytes of a real
            // buffer, which is worse than reading a stale copy: it would corrupt llama's cache.
            if (t->type != r.cache_type || (size_t) ggml_nbytes(t) != want) {
                fprintf(stderr, "[tsi-mlir] cache_%s_l%d is %s %zu bytes, expected %s %zu\n", k, il,
                        ggml_type_name(t->type), (size_t) ggml_nbytes(t),
                        ggml_type_name(r.cache_type), want);
                return false;
            }
            out.push_back(t->data);
        }
    }
    return true;
}


// llama's live attention window (n_kv), read off the 3-d VIEW a decode step takes over cache_k.
//
// One compiled binary is built for one window. llama grows n_kv in blocks as the context fills, and
// when it does the graph shape, the binary and the cache size all change together.
int liveWindow(ggml_cgraph * live) {
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_VIEW && nd->src[0] && ggml_n_dims(nd) == 3 &&
            wg_core_name(nd->src[0]->name).rfind("cache_k_l", 0) == 0) {
            return (int) nd->ne[2];
        }
    }
    return 0;
}

// One compiled decode, reused for every token of a generation.
//
// Built once and kept: the rebuilt ggml graph, the compiled binary, and the argv - including the two
// cache descriptors, which point at buffers that outlive every call. Per token only id, pos, mask and
// slot change, and all four are runtime arguments, so a single binary serves every position. Rebuilding
// and re-exporting per token would otherwise dominate: it re-reads every weight and re-hashes the
// module for a graph that is structurally identical.
struct DecodeSession {
    decode_case     r;
    forward_argv_fn fwd       = nullptr;
    DeviceArgs          args;
    std::vector<void *> cache_ptrs;    // llama's buffers, k layers then v layers
    size_t              layer_bytes = 0;
    int64_t         nvoc      = 0;
    int             cells     = 0;   // the window this binary was compiled for
    int             steps     = 0;
    int             last_pos  = -1;  // to catch a context reset or a new sequence
    bool            ok        = false;

    ~DecodeSession() {
        if (r.ctx) {
            ggml_free(r.ctx);
        }
        if (r.wc) {
            ggml_free(r.wc);
        }
    }
};

// Deliberately a raw pointer that is never freed at exit. Its destructor would tsi_dealloc device
// buffers, and static destruction can run after the atexit tsi_finalize in Runtime.h, which would be a
// use-after-teardown. Deleted explicitly when the window changes, which happens while the runtime is
// still up.
DecodeSession * g_decode = nullptr;

// Build, compile, seed. Returns nullptr on any failure, in which case llama's own decode stands.
DecodeSession * openDecodeSession(ggml_cgraph * live, const Config & cfg) {
    auto * s = new DecodeSession();
    try {
        s->r = build_decode_from_live(live, cfg.weight_args);
    } catch (const std::exception & e) {
        fprintf(stderr, "[tsi-mlir] decode SKIPPED: %s\n", e.what());
        delete s;
        return nullptr;
    }

    // A separate artifact from prefill: different arity, so a separate RTLD_LOCAL handle too.
    s->fwd = buildForward(s->r.func_text, "decode", cfg);
    if (!s->fwd) {
        delete s;
        return nullptr;
    }
    // The module is compiled; on a weights-baked run it is hundreds of MiB of bytecode with nothing
    // left to read it.
    s->r.func_text.clear();
    s->r.func_text.shrink_to_fit();

    s->nvoc  = s->r.outputs[0]->ne[0];
    s->cells = s->r.cells;

    runtimeUp();
    if (!collectCachePtrs(live, s->r, s->cache_ptrs)) {
        delete s;
        return nullptr;
    }
    s->layer_bytes = (size_t) s->r.cache_shape[1] * s->r.cache_shape[2] * s->r.cache_shape[3] *
                     ggml_type_size(s->r.cache_type);

    // argv order is the ciface order the exporter documents:
    // [runtime_args..., cache memrefs..., outputs...]. No `slot`: nothing is appended, so the exporter
    // emits no cell index at all.
    bool ok = true;
    for (const ggml_tensor * t : s->r.runtime_args) {
        ok = ok && s->args.addInput(t);
    }
    for (void * p : s->cache_ptrs) {
        ok = ok && s->args.addCache(p, s->r.cache_shape);
    }
    for (const ggml_tensor * t : s->r.outputs) {
        ok = ok && s->args.addOutput(t);
    }
    if (!ok) {
        delete s;
        return nullptr;
    }

    fprintf(stderr, "[tsi-mlir] decode session ready: %zu args + %zu cache memrefs, %d of %d cells, "
                    "%s cache aliased in place (llama owns the write), %zu results\n",
            s->r.runtime_args.size(), s->cache_ptrs.size(), s->cells, s->r.capacity,
            ggml_type_name(s->r.cache_type), s->r.outputs.size());
    s->ok = true;
    return s;
}

// Write this step's K/V into llama's cache at the cell llama chose.
//
// Needed only because llama's forward pass is skipped, which skips its SET_ROWS along with it. The
// single-cell case of writeCacheCells, which prefill uses with n_tokens cells.
bool writeCacheFromResults(DecodeSession & s, ggml_cgraph * live, const std::vector<int> & cells) {
    if (cells.size() != 1) {
        fprintf(stderr, "[tsi-mlir] decode expects one cell, got %zu\n", cells.size());
        return false;
    }
    return writeCacheCells(live, s.args, s.r.n_layers, cells);
}

// Run one token through the compiled decode. Only the per-step inputs move host->device; the cache is
// read where it lives and never copied.
std::vector<float> runDecodeStep(DecodeSession & s, const Config & cfg, int64_t & nvoc_out) {
    nvoc_out = s.nvoc;

    if (g_ids_cap.size() != 1 || g_pos_cap.size() != 1) {
        return {};
    }
    if (!decode_retarget(s.r, g_ids_cap[0], g_pos_cap[0])) {
        fprintf(stderr, "[tsi-mlir] position %d is past the %d-cell window; decode SKIPPED\n",
                g_pos_cap[0], s.cells);
        return {};
    }
    // id, pos and mask, in the order build_decode_from_live declared them. A few KiB per token, versus
    // the whole cache before.
    for (size_t k = 0; k < 3; k++) {
        s.args.refreshInput(k, s.r.runtime_args[k]);
    }

    // The CPU reference reads the f32 cache snapshot taken when the session opened, and nothing
    // updates it: the live cache now advances on the device. So it is only meaningful on the first
    // step, and running it later would compare against a stale state and look like a regression.
    std::vector<float> reconcpu;
    const bool         want_cpu_ref = cfg.cpu_ref && s.steps == 0;
    if (want_cpu_ref) {
        reconcpu.resize((size_t) s.nvoc);
        ggml_graph_compute_with_ctx(s.r.ctx, s.r.gf, 4);
        memcpy(reconcpu.data(), s.r.outputs[0]->data, (size_t) s.nvoc * sizeof(float));
    } else if (cfg.cpu_ref && s.steps == 1) {
        fprintf(stderr, "[tsi-mlir] decode CPU reference is first-step only; the cache it would need "
                        "now lives on the device\n");
    }

    s.fwd(s.args.argv());
    s.steps++;
    s.last_pos = s.r.pos;

    std::vector<float> compiled((size_t) s.nvoc);
    memcpy(compiled.data(), s.args.output(0), (size_t) s.nvoc * sizeof(float));

    // TSI_MLIR_CACHE_SUM=1: fingerprint the device cache and the logits after every step.
    //
    // Diagnostic for divergence that only appears after several tokens: comparing two runs step by
    // step says whether the cache or the logits moved first, and at which token. The cache is the only
    // state carried between calls, so if it matches and the logits do not, the compiled body is at
    // fault rather than the cache handling.
    if (getenv("TSI_MLIR_CACHE_SUM")) {
        auto fnv = [](const void * p, size_t n) {
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 0; i < n; i++) {
                h = (h ^ ((const unsigned char *) p)[i]) * 1099511628211ull;
            }
            return h;
        };
        // Layer 0 only, so a step does not hash 90 MiB. Enough to catch a cache that moved.
        fprintf(stderr, "[tsi-mlir] step %2d pos %3d  k0=%016llx v0=%016llx logits=%016llx\n",
                s.steps - 1, s.r.pos,
                (unsigned long long) fnv(s.cache_ptrs.front(), s.layer_bytes),
                (unsigned long long) fnv(s.cache_ptrs[(size_t) s.r.n_layers], s.layer_bytes),
                (unsigned long long) fnv(compiled.data(), compiled.size() * sizeof(float)));
    }

    if (want_cpu_ref) {
        compare("decode recon-CPU vs compiled:", reconcpu.data(), compiled.data(), s.nvoc);
    }
    return compiled;
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

bool tsi_mlir_export_before_compute(struct ggml_cgraph * cgraph) {
    const Config & cfg = Config::get();
    if (!cfg.enabled) {
        return false;
    }
    // Each graph gets fresh snapshots: ids and positions differ per step, and a stale position would
    // misclassify the phase.
    g_ids_cap.clear();
    g_pos_cap.clear();

    // Positions AND token ids are graph INPUTS, filled by llama before compute, so read them here
    // rather than waiting for the eval callback. That matters because the phase decides whether we need a cache
    // snapshot, and the snapshot has to happen now: after compute, llama has already written this
    // step's cell and a decode graph would be reading its own answer.
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
        ggml_tensor * nd = ggml_graph_node(cgraph, i);
        if (nd->op == GGML_OP_ROPE && nd->src[1] && nd->src[1]->type == GGML_TYPE_I32 &&
            nd->src[1]->data) {
            g_pos_cap.resize((size_t) ggml_nelements(nd->src[1]));
            memcpy(g_pos_cap.data(), nd->src[1]->data, g_pos_cap.size() * sizeof(int32_t));
            break;
        }
    }

    // The decode token id. Same reason as positions: needed before compute, and the eval callback
    // has not run yet. Located structurally as the i32 src1 of the GET_ROWS over the embedding table,
    // because llama names this input differently across versions.
    //
    // Reading it from g_ids_cap instead was a real bug: that is filled during compute and cleared
    // just above, so decode ran on token id 0 and produced confidently wrong logits.
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); i++) {
        ggml_tensor * nd = ggml_graph_node(cgraph, i);
        if (nd->op == GGML_OP_GET_ROWS && nd->src[0] && nd->src[1] &&
            wg_core_name(nd->src[0]->name) == "token_embd.weight" &&
            nd->src[1]->type == GGML_TYPE_I32 && nd->src[1]->data) {
            g_ids_cap.resize((size_t) ggml_nelements(nd->src[1]));
            memcpy(g_ids_cap.data(), nd->src[1]->data, g_ids_cap.size() * sizeof(int32_t));
            break;
        }
    }

    static int seen      = 0;
    const bool is_warmup = seen++ < cfg.skip;

    // Decode is handled HERE, not after compute. The graph reads cells 0..pos-1, and after compute
    // llama has written cell pos, so a graph built then would consume its own answer.
    //
    // EVERY decode token runs compiled, not just the first. The session holds one binary and one
    // device-resident cache, so a token costs one call plus a few KiB of inputs. This used to be
    // guarded by a one-shot flag, which meant token 1 ran on TSI and every later token silently fell
    // back to llama - the compiled decode path was never actually exercised for a generation.
    bool handled = false;

    if (!is_warmup && classify() == Phase::Decode) {
        const int win = liveWindow(cgraph);
        const int pos = g_pos_cap.empty() ? -1 : g_pos_cap[0];

        if (g_decode && win > 0 && g_decode->cells != win) {
            // llama grew its attention window, so the graph shape, the binary and the cache size all
            // change. Rebuild rather than reuse a binary built for a smaller window.
            fprintf(stderr, "[tsi-mlir] llama's window grew %d -> %d cells; rebuilding decode\n",
                    g_decode->cells, win);
            delete g_decode;
            g_decode = nullptr;
        }
        if (g_decode && pos != g_decode->last_pos + 1) {
            // Positions are meant to advance by one. Anything else means llama restarted or switched
            // sequence - a conversation reset, a context shift, a cleared slot - and the graph's mask
            // and cached geometry no longer describe the state. Rebuilding re-reads everything from the
            // live graph, which is correct by construction; carrying on would compute against the
            // previous sequence's keys and report nothing.
            fprintf(stderr, "[tsi-mlir] position jumped %d -> %d; rebuilding decode\n",
                    g_decode->last_pos, pos);
            delete g_decode;
            g_decode = nullptr;
        }
        if (!g_decode) {
            reportDecodeCache(cgraph);
            g_decode = openDecodeSession(cgraph, cfg);
        }
        g_decode_logits = g_decode ? runDecodeStep(*g_decode, cfg, g_decode_nvoc)
                                   : std::vector<float>();

        // Take over the step only if the cache was updated too. Under TSI_MLIR_VERIFY llama has to
        // compute in order to be the reference, so it keeps its own SET_ROWS and we write nothing -
        // otherwise both would write the same cell and the comparison would be measuring itself.
        if (!g_decode_logits.empty() && !cfg.verify) {
            std::vector<int> cells;
            handled = liveCacheCells(cgraph, cells) &&
                      writeCacheFromResults(*g_decode, cgraph, cells);
            if (!handled) {
                // Leave llama to compute: a half-updated cache is worse than a duplicated pass.
                fprintf(stderr, "[tsi-mlir] could not update llama's cache; falling back to llama "
                                "for this token\n");
            }
        }
    }
    return handled;
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

    // Prefill happens once. Decode repeats per token and every one of them is run compiled, in
    // before_compute; this only writes the result back.
    static bool did_prefill = false;

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

    if (phase == Phase::Decode && !g_decode_logits.empty()) {
        ggml_tensor * live_out = ggml_graph_node(live, -1);
        const int64_t n = ggml_nelements(live_out) < g_decode_nvoc ? ggml_nelements(live_out)
                                                                  : g_decode_nvoc;
        if (cfg.verify) {
            compare("decode compiled vs llama:", g_decode_logits.data(),
                    (const float *) live_out->data, n);
        }
        memcpy(live_out->data, g_decode_logits.data(), (size_t) n * sizeof(float));
        g_decode_logits.clear();
    }
    return false;
}
