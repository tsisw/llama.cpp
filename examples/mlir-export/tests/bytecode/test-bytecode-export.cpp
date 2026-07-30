// Checks the two things weights-as-constants depends on: that an undeclared leaf is baked in as a
// dense_resource constant, and that bytecode is a working, smaller encoding of the same module.
//
// Size is the whole reason bytecode exists. A blob prints as hex, two characters per byte, so the
// text form of a real model is twice its weight bytes; bytecode stores them raw. The assertion below
// pins that down on a graph small enough to eyeball.
//
// The emitted <dir>/forward.mlirbc is what the companion ctest entry feeds to the compiler, which is
// the only check that matters in the end: the compiler has to accept what we produce.
//
// Usage: test-bytecode-export <out-dir>
#include "tsi/export/Exporter.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        failures++;
    }
}

// Writes `bytes` and returns false if anything went wrong. Binary mode: bytecode contains NULs, and
// an ofstream to a missing directory fails silently, which would report success having written
// nothing.
bool write_file(const std::filesystem::path & path, const std::string & bytes) {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), (std::streamsize) bytes.size());
    f.close();
    return (bool) f;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out-dir>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path dir(argv[1]);

    const int64_t N = 32, K = 64;

    ggml_init_params ip { (size_t) 16 << 20, nullptr, false };
    ggml_context *   ctx = ggml_init(ip);

    // x is the per-step input; w is the weight. Only x is declared, so w gets baked.
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(x, "x");
    ggml_set_name(w, "proj.weight");
    for (int64_t i = 0; i < ggml_nelements(w); i++) {
        ((float *) w->data)[i] = (float) (i % 17) * 0.25f;
        ((float *) x->data)[i] = (float) (i % 7) * 0.5f;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));

    tsi::mlir_export::ExportOptions opts;
    opts.runtime_args = { x };

    std::string text, bc;
    try {
        opts.format = tsi::mlir_export::Format::Text;
        text        = tsi::mlir_export::exportGraph(gf, opts);
        opts.format = tsi::mlir_export::Format::Bytecode;
        bc          = tsi::mlir_export::exportGraph(gf, opts);
    } catch (const std::exception & e) {
        fprintf(stderr, "export failed: %s\n", e.what());
        ggml_free(ctx);
        return 1;
    }

    // The weight is a resource-backed constant, not an inline dense<> literal.
    check(text.find("dense_resource<proj.weight>") != std::string::npos,
          "the undeclared leaf is baked as dense_resource<proj.weight>");
    check(text.find("dialect_resources") != std::string::npos,
          "text carries the blob in a dialect_resources section");
    // One argument, so the weight is gone from the signature - the point of the whole exercise.
    check(text.find("%arg1") == std::string::npos, "only the declared leaf is an argument");

    check(bc.size() >= 4 && memcmp(bc.data(), "ML\xef" "R", 4) == 0,
          "bytecode starts with the MLIR magic bytes");
    check(bc.find(std::string("dense_resource")) == std::string::npos,
          "bytecode has no textual attribute spelling");

    // Hex is 2 bytes per data byte plus the "0x" and the alignment header, so bytecode must come in
    // under the text form by at least the weight's own size.
    const size_t wbytes = (size_t) ggml_nbytes(w);
    printf("     text %zu bytes, bytecode %zu bytes, weight %zu bytes\n", text.size(), bc.size(),
           wbytes);
    check(bc.size() + wbytes < text.size(), "bytecode is smaller than text by at least the blob");

    check(write_file(dir / "forward.mlir", text), "wrote forward.mlir");
    check(write_file(dir / "forward.mlirbc", bc), "wrote forward.mlirbc");

    ggml_free(ctx);
    printf("%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
