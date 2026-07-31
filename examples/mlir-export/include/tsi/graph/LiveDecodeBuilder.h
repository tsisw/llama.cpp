#pragma once
// Rebuild a decode graph from llama.cpp's live forward cgraph, so the compiled decode runs on the
// state llama is actually in.
//
// The counterpart of LiveGraphBuilder.h, which does the same for prefill. Neither translates llama's
// graph: llama uses FLASH_ATTN_EXT (nothing equivalent in our lowering) and mutates its cache through
// SET_ROWS, which a pure tensor function cannot express. So the math is rebuilt from primitives we do
// lower, via build_decode in DecodeModel.h - the same builder decode_cpu_check validates on CPU, so
// the checked graph and the exported one cannot drift apart.
//
// The cache is a pair of DRAM memrefs written in place (see CacheSpec in Exporter.h), so the
// signature is {id, pos, mask} + cache_k + cache_v + slot -> logits: 6 arguments and one result,
// independent of layer count. It went through an intermediate form first, with one tensor argument
// per layer per kind (63 arguments, 61 results on SmolLM2), which is why the numbers in the git
// history jump: that form proved the decode math produced the right logits, and this one is a
// signature change on top of an already-correct path rather than two unknowns at once.
//
// The host still seeds the DRAM buffers from llama's live cache on every step. Letting the compiled
// prefill fill them instead is the next step, and it is the one that proves the cache round-trips
// through DRAM rather than being re-supplied each time.
#include "tsi/graph/DecodeModel.h"      // DecodeModel, build_decode
#include "tsi/graph/LiveCache.h"        // live_cache_probe / live_cache_extract
#include "tsi/graph/LiveGraphBuilder.h" // wg_core_name, wg_index_live, case_result, g_wcap

#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// A rebuilt decode step: the graph, its arguments and outputs in the exporter's order, and the MLIR.
struct decode_case {
    struct ggml_context * ctx = nullptr;
    struct ggml_cgraph  * gf  = nullptr;
    // Holds the staged weight copies. Kept alive because gf still points at them: freeing it would
    // leave the graph with dangling weights, which only bites whoever computes it on CPU later.
    struct ggml_context * wc  = nullptr;

    // [id, pos, mask]. The caches and the slot follow in the ciface argument order, but they are not
    // ggml tensors, so they are described below instead.
    std::vector<const ggml_tensor *> runtime_args;
    // [logits]. K/V are no longer results: the graph appends them into the cache memrefs in place.
    std::vector<const ggml_tensor *> outputs;

    // The two DRAM caches, in argument order after runtime_args: cache_k then cache_v, then `slot`.
    //
    // Stored in llama's own cache type, so the device buffer is byte-compatible with llama's and can be
    // seeded by a plain memcpy. The graph still computes in f32; the exporter widens on read and
    // narrows on append.
    ggml_type            cache_type  = GGML_TYPE_F16;
    std::vector<int64_t> cache_shape;        // MLIR order: [n_layers, cells, n_head_kv, head_dim]
    size_t               cache_bytes = 0;    // one cache, all layers
    size_t               layer_bytes = 0;    // one layer's window, for seeding layer by layer

    std::string func_text;
    int         pos   = 0;   // position of the token being decoded; also the cache slot to append at
    int         cells = 0;   // cache window this graph was built for
};

// Fill one step's inputs: which token, which position, and the mask that gates the cache.
//
// The mask rule is the subtle part, so it lives here alone and both the initial build and every
// subsequent token go through it. Cells 0..pos-1 hold real keys; cells from pos on have not been
// written yet and must be masked off, or uninitialized cells read as context. The final slot is the new
// token attending to itself.
static inline void ldb_fill_step(ggml_tensor * id_t, ggml_tensor * pos_t, ggml_tensor * mask_t,
                                 int cells, int32_t token, int pos) {
    ((int32_t *) id_t->data)[0]  = token;
    ((int32_t *) pos_t->data)[0] = pos;

    float * m = (float *) mask_t->data;
    for (int j = 0; j < cells; j++) {
        m[j] = j < pos ? 0.0f : -INFINITY;
    }
    m[cells] = 0.0f;
}

// Read one i32/f32 op_param slot, the way ggml stores them.
static inline int32_t ldb_pi32(const ggml_tensor * t, int slot) {
    return ((const int32_t *) t->op_params)[slot];
}
static inline float ldb_pf32(const ggml_tensor * t, int slot) {
    float v;
    memcpy(&v, (const char *) t->op_params + slot * sizeof(int32_t), sizeof(float));
    return v;
}

// Rebuild the decode step this live graph represents.
//
// MUST be called before llama computes the graph: the cache read has to see cells 0..pos-1, and after
// compute llama has already written cell pos, so the graph would be consuming its own answer.
//
// Throws mlir_export_error when the live graph is not a decode step we can express.
static inline decode_case build_decode_from_live(struct ggml_cgraph * live,
                                                 bool weights_as_args = false) {
    using tsi::mlir_export::mlir_export_error;

    std::map<std::string, ggml_tensor *> idx = wg_index_live(live);

    auto need = [&](const std::string & core) -> ggml_tensor * {
        auto it = idx.find(core);
        if (it == idx.end() || !it->second) {
            throw mlir_export_error("live graph missing tensor '" + core + "'");
        }
        return it->second;
    };
    auto want = [&](const std::string & core) -> ggml_tensor * {
        auto it = idx.find(core);
        return it == idx.end() ? nullptr : it->second;
    };

    // --- geometry, straight off the live graph ------------------------------------------------
    ggml_tensor * embd_t = need("token_embd.weight");
    DecodeModel   M;
    M.hidden  = (int) embd_t->ne[0];
    M.n_vocab = (int) embd_t->ne[1];

    while (idx.count("blk." + std::to_string(M.n_layers) + ".attn_q.weight")) {
        M.n_layers++;
    }
    if (M.n_layers == 0) {
        throw mlir_export_error("live graph has no blk.*.attn_q.weight layers");
    }
    M.inter = (int) need("blk.0.ffn_gate.weight")->ne[1];

    // head_dim and the rope config come from the first ROPE node: src0 is [head_dim, n_head, n_tok].
    ggml_tensor * rope = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(live) && !rope; i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_ROPE && nd->src[0] && nd->src[1]) {
            rope = nd;
        }
    }
    if (!rope) {
        throw mlir_export_error("live graph has no ROPE node");
    }
    M.head_dim  = (int) rope->src[0]->ne[0];
    M.n_head    = (int) need("blk.0.attn_q.weight")->ne[1] / M.head_dim;
    M.n_head_kv = (int) need("blk.0.attn_k.weight")->ne[1] / M.head_dim;
    M.hidden_kv = M.head_dim * M.n_head_kv;

    REAL_ROPE             = RopeParams{};
    REAL_ROPE.n_dims      = ldb_pi32(rope, 1);
    REAL_ROPE.mode        = ldb_pi32(rope, 2);
    REAL_ROPE.n_ctx_orig  = ldb_pi32(rope, 4);
    REAL_ROPE.freq_base   = ldb_pf32(rope, 5);
    REAL_ROPE.freq_scale  = ldb_pf32(rope, 6);
    REAL_ROPE.ext_factor  = ldb_pf32(rope, 7);
    REAL_ROPE.attn_factor = ldb_pf32(rope, 8);
    REAL_ROPE.beta_fast   = ldb_pf32(rope, 9);
    REAL_ROPE.beta_slow   = ldb_pf32(rope, 10);

    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_RMS_NORM) {
            memcpy(&REAL_RMS_EPS, nd->op_params, sizeof(float));
            break;
        }
    }
    // FLASH_ATTN_EXT carries the scale in op_params[0], the same slot SOFT_MAX uses in the unfused
    // form. Falling back to 1/sqrt(head_dim) if absent, which is what build_decode_layer assumes.
    REAL_KQ_SCALE = 1.0f / sqrtf((float) M.head_dim);
    for (int i = 0; i < ggml_graph_n_nodes(live); i++) {
        ggml_tensor * nd = ggml_graph_node(live, i);
        if (nd->op == GGML_OP_FLASH_ATTN_EXT || nd->op == GGML_OP_SOFT_MAX) {
            memcpy(&REAL_KQ_SCALE, nd->op_params, sizeof(float));
            break;
        }
    }

    // --- the cache -----------------------------------------------------------------------------
    const LiveCacheInfo ck_info = live_cache_probe(live, "k", M.head_dim, M.n_head_kv);
    const LiveCacheInfo cv_info = live_cache_probe(live, "v", M.head_dim, M.n_head_kv);
    if (!ck_info.found || !cv_info.found) {
        throw mlir_export_error("live graph has no cache_k_l*/cache_v_l* buffers");
    }
    // Match llama's live window rather than the full n_ctx: one compiled binary per window size, and
    // llama only ever attends over n_kv cells.
    const int L = ck_info.n_kv > 0 ? ck_info.n_kv : ck_info.n_cells;

    if (g_pos_cap.empty()) {
        throw mlir_export_error("no positions captured; cannot tell which cell to write");
    }
    const int pos = g_pos_cap[0];
    if (g_pos_cap.size() != 1) {
        throw mlir_export_error("decode expects a single token, got " +
                                std::to_string(g_pos_cap.size()));
    }
    if (pos >= L) {
        throw mlir_export_error("position " + std::to_string(pos) + " is outside the " +
                                std::to_string(L) + "-cell window this graph attends over");
    }

    // --- weights -------------------------------------------------------------------------------
    // Sized from the weights actually seen, doubled for activations, as the prefill path does. A
    // fixed guess aborts on a real model with "GGML_ASSERT(obj_new) failed".
    size_t wbytes = 0;
    for (auto & kv : idx) {
        if (kv.second && kv.second->op == GGML_OP_NONE && kv.second->data) {
            wbytes += ggml_nbytes(kv.second);
        }
    }
    ggml_init_params wp { wbytes * 2 + ((size_t) 128 << 20), nullptr, false };
    M.wc = ggml_init(wp);

    // Copy a weight into a fresh contiguous f32 tensor, preferring the snapshot taken while llama
    // computed: the scheduler's copies are recycled scratch by now.
    auto bind = [&](const std::string & core, ggml_tensor * shape_src) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor(M.wc, GGML_TYPE_F32, ggml_n_dims(shape_src), shape_src->ne);
        auto          it = g_wcap.find(core);
        if (it != g_wcap.end() && it->second.size() == (size_t) ggml_nelements(t)) {
            memcpy(t->data, it->second.data(), ggml_nbytes(t));
        } else if (shape_src->data && shape_src->type == GGML_TYPE_F32) {
            memcpy(t->data, shape_src->data, ggml_nbytes(t));
        } else {
            throw mlir_export_error("no usable data for weight '" + core + "'");
        }
        ggml_set_name(t, core.c_str());
        return t;
    };

    M.embd = bind("token_embd.weight", embd_t);
    M.onw  = bind("output_norm.weight", need("output_norm.weight"));
    // Tied embeddings (SmolLM2, Llama 3.2 1B): no separate output projection, reuse the embedding
    // matrix. Re-binding "output.weight" would fall back to recycled scratch and read garbage.
    ggml_tensor * oww_src = want("output.weight");
    M.oww                 = oww_src ? bind("output.weight", oww_src) : M.embd;

    M.lw.resize(M.n_layers);
    for (int il = 0; il < M.n_layers; il++) {
        const std::string p = "blk." + std::to_string(il) + ".";
        M.lw[il].attn_norm  = bind(p + "attn_norm.weight",   need(p + "attn_norm.weight"));
        M.lw[il].wq         = bind(p + "attn_q.weight",      need(p + "attn_q.weight"));
        M.lw[il].wk         = bind(p + "attn_k.weight",      need(p + "attn_k.weight"));
        M.lw[il].wv         = bind(p + "attn_v.weight",      need(p + "attn_v.weight"));
        M.lw[il].wo         = bind(p + "attn_output.weight", need(p + "attn_output.weight"));
        M.lw[il].ffn_norm   = bind(p + "ffn_norm.weight",    need(p + "ffn_norm.weight"));
        M.lw[il].gate       = bind(p + "ffn_gate.weight",    need(p + "ffn_gate.weight"));
        M.lw[il].up         = bind(p + "ffn_up.weight",      need(p + "ffn_up.weight"));
        M.lw[il].down       = bind(p + "ffn_down.weight",    need(p + "ffn_down.weight"));
    }

    // --- the step's inputs ---------------------------------------------------------------------
    decode_case r;
    r.pos   = pos;
    r.cells = L;

    // Sized for the graph plus the cache copies: 2 * n_layers windows of head_dim*n_head_kv*L floats.
    const size_t cache_bytes =
        (size_t) 2 * M.n_layers * M.head_dim * M.n_head_kv * L * sizeof(float);
    ggml_init_params gp { cache_bytes * 2 + ((size_t) 512 << 20), nullptr, false };
    r.ctx = ggml_init(gp);

    ggml_tensor * id = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 1);
    ggml_set_name(id, "id");
    if (g_ids_cap.size() != 1) {
        // Silently defaulting here is how this produced confidently wrong logits the first time: the
        // graph compiles and runs, and only the numbers say anything is amiss.
        throw mlir_export_error("decode expects exactly one token id, got " +
                                std::to_string(g_ids_cap.size()));
    }

    ggml_tensor * pt = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 1);
    ggml_set_name(pt, "pos");

    // mask[0..L-1] gates the cache cells, mask[L] is the new token attending to itself.
    ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, L + 1, 1);
    ggml_set_name(mask, "mask");

    ldb_fill_step(id, pt, mask, L, g_ids_cap[0], pos);

    std::vector<ggml_tensor *> cK(M.n_layers), cV(M.n_layers);
    for (int il = 0; il < M.n_layers; il++) {
        cK[il] = live_cache_extract(r.ctx, live, "k", il, ck_info, L);
        cV[il] = live_cache_extract(r.ctx, live, "v", il, cv_info, L);
        if (!cK[il] || !cV[il]) {
            throw mlir_export_error("could not read llama's cache for layer " + std::to_string(il));
        }
        ggml_set_name(cK[il], ("cache_k_" + std::to_string(il)).c_str());
        ggml_set_name(cV[il], ("cache_v_" + std::to_string(il)).c_str());
    }

    // --- build and export -----------------------------------------------------------------------
    std::vector<ggml_tensor *> knew, vnew;
    ggml_tensor *              logits = build_decode(r.ctx, M, id, pt, mask, cK, cV, knew, vnew);

    r.gf = ggml_new_graph_custom(r.ctx, 16384, false);
    ggml_build_forward_expand(r.gf, logits);
    for (int il = 0; il < M.n_layers; il++) {
        ggml_build_forward_expand(r.gf, knew[il]);
        ggml_build_forward_expand(r.gf, vnew[il]);
    }

    // Only the per-step inputs are arguments. The cache leafs become reads of the memref, and
    // knew/vnew become appends into it, so neither appears in the signature.
    r.runtime_args = { id, pt, mask };
    r.outputs      = { logits };

    // Weights as arguments instead of baked constants, when the constant pool would be too big to
    // land in one object file. See Config::weight_args. The cache leafs stay out of this: they are
    // caches, not arguments, and adding them here would emit each layer's window twice.
    if (weights_as_args) {
        std::set<const ggml_tensor *> excl(r.runtime_args.begin(), r.runtime_args.end());
        excl.insert(cK.begin(), cK.end());
        excl.insert(cV.begin(), cV.end());
        for (const ggml_tensor * leaf : tsi::mlir_export::discoverLeafs(r.gf)) {
            if (!excl.count(leaf)) {
                r.runtime_args.push_back(leaf);
            }
        }
    }

    // Store the cache in whatever type llama stores its own in, so seeding is a memcpy and a value we
    // append rounds to the pattern llama would have written. Falls back to f32 if llama's cache is f32.
    ggml_tensor * k_leaf = live_cache_leaf(live, "k", 0);
    r.cache_type         = k_leaf ? k_leaf->type : GGML_TYPE_F16;
    if (r.cache_type != GGML_TYPE_F16 && r.cache_type != GGML_TYPE_F32) {
        throw mlir_export_error(std::string("llama's KV cache is ") +
                                ggml_type_name(r.cache_type) + ", which has no memref element type");
    }

    // MLIR order, [n_layers, cells, n_head_kv, head_dim]: ggml's ne reversed, with the layer dim
    // prepended. Must match GraphBuilder::cacheType or the descriptor strides address the wrong cell.
    r.cache_shape = { M.n_layers, L, M.n_head_kv, M.head_dim };
    r.layer_bytes = (size_t) L * M.n_head_kv * M.head_dim * ggml_type_size(r.cache_type);
    r.cache_bytes = (size_t) M.n_layers * r.layer_bytes;

    tsi::mlir_export::ExportOptions opts;
    opts.runtime_args = r.runtime_args;
    opts.outputs      = r.outputs;

    // One CacheSpec per kind. `read` is the leaf the graph consumes, `append` the value computed for
    // this step; the exporter substitutes a read of layer il for the former and stores the latter at
    // `slot`. f32 throughout: the graph computes in f32 and the host seeds these buffers from its own
    // f32 conversion, so there is nothing to narrow yet. Making the cache f16 to match llama bit for
    // bit is a size/fidelity change that belongs with the prefill handoff, not here.
    tsi::mlir_export::CacheSpec ks, vs;
    ks.name      = "cache_k";
    ks.n_layers  = M.n_layers;
    ks.cells     = L;
    ks.elem_type = r.cache_type;
    ks.read.assign(cK.begin(), cK.end());
    ks.append.assign(knew.begin(), knew.end());
    vs.name      = "cache_v";
    vs.n_layers  = M.n_layers;
    vs.cells     = L;
    vs.elem_type = r.cache_type;
    vs.read.assign(cV.begin(), cV.end());
    vs.append.assign(vnew.begin(), vnew.end());
    opts.caches = { ks, vs };

    // Bytecode: the weights are baked in as constants, and text would hex-print them at twice the size.
    opts.format = tsi::mlir_export::Format::Bytecode;
    r.func_text = tsi::mlir_export::exportGraph(r.gf, opts);

    fprintf(stderr, "[tsi-mlir] decode rebuilt: pos %d, %d cells, %zu args + 2 caches + slot, "
                    "%zu outputs, %.2f MiB cache each, %.2f MiB bytecode\n",
            pos, L, r.runtime_args.size(), r.outputs.size(),
            (double) r.cache_bytes / (1024.0 * 1024.0),
            (double) r.func_text.size() / (1024.0 * 1024.0));

    r.wc = M.wc;   // freed by the caller, together with r.ctx
    return r;
}

// Retarget an already-built decode_case at the next token, so one compiled binary serves a whole
// generation.
//
// id, pos and mask are runtime arguments and the graph is structurally identical from one token to the
// next, so only their data moves. This is what makes per-token compiled decode cheap: no rebuild, no
// re-export, no recompile, and the KV cache stays on the device.
//
// `runtime_args` is [id, pos, mask] by construction above, and the driver's argv depends on that order.
//
// Returns false when the position has run past the window this graph was built for. That needs a new
// graph, not a new mask, so it is the caller's cue to rebuild rather than something to paper over.
static inline bool decode_retarget(decode_case & r, int32_t token, int pos) {
    if (pos < 0 || pos >= r.cells || r.runtime_args.size() < 3) {
        return false;
    }
    ldb_fill_step(const_cast<ggml_tensor *>(r.runtime_args[0]),
                  const_cast<ggml_tensor *>(r.runtime_args[1]),
                  const_cast<ggml_tensor *>(r.runtime_args[2]), r.cells, token, pos);
    r.pos = pos;
    return true;
}
