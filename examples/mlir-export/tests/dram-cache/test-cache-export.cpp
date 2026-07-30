// Emits a minimal graph that reads a DRAM cache and appends to it, to check the exporter's cache
// path produces valid, compilable MLIR.
//
// Shapes are tiny but the structure is the real one: one memref argument for the cache, a scalar
// slot, a read of the whole window feeding the body, and an append after it.
//
// MLIR shape is ggml's ne reversed, and the cache wants cells as its outermost slice dim, so a slice
// is built with cells LAST in ggml terms: ggml_new_tensor_3d(ctx, F32, head_dim, n_head_kv, cells)
// becomes memref<... x cells x n_head_kv x head_dim>. Same for the appended value, whose cell count
// is 1 for a decode step.
//
// Usage: test-cache-export <out.mlir>
#include "tsi/export/Exporter.h"
#include "ggml.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.mlir>\n", argv[0]);
        return 2;
    }

    const int64_t HEAD_DIM = 4, N_HEAD_KV = 2, CELLS = 8, N_NEW = 1;

    ggml_init_params ip { (size_t) 16 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // Stands for one layer's cache window. Never a real argument: the exporter replaces it with a
    // read of the memref.
    ggml_tensor * slice = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV, CELLS);
    ggml_set_name(slice, "cache_slice");

    // The token's K/V, and the value actually appended (a node, so valueOf resolves a computed value).
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEAD_KV, N_NEW);
    ggml_set_name(x, "k_new");
    ggml_tensor * appended = ggml_scale(ctx, x, 3.0f);

    // Something that consumes the cache read, so the read cannot be dead-code eliminated.
    ggml_tensor * out = ggml_scale(ctx, slice, 2.0f);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, appended);

    tsi::mlir_export::CacheSpec cache;
    cache.name     = "cache_k";
    cache.n_layers = 1;
    cache.cells    = CELLS;
    cache.read     = { slice };
    cache.append   = { appended };

    tsi::mlir_export::ExportOptions opts;
    opts.runtime_args = { x };
    opts.outputs      = { out };
    opts.caches       = { cache };

    std::string mlir;
    try {
        mlir = tsi::mlir_export::exportGraph(gf, opts);
    } catch (const std::exception & e) {
        fprintf(stderr, "export failed: %s\n", e.what());
        ggml_free(ctx);
        return 1;
    }

    // Create the parent directory and check the write. An ofstream to a missing directory fails
    // silently, which would report success while producing nothing.
    const std::filesystem::path path(argv[1]);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream f(path);
    f << mlir;
    f.close();
    if (!f) {
        fprintf(stderr, "failed to write %s\n", argv[1]);
        ggml_free(ctx);
        return 1;
    }

    printf("emitted cache export (%zu bytes) -> %s\n", mlir.size(), argv[1]);
    ggml_free(ctx);
    return 0;
}
