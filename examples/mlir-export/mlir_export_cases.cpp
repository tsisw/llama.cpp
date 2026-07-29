// Emits self-contained test cases for the ggml -> linalg MLIR exporter.
//
// Per case: build a small ggml graph, fill its inputs from a fixed seed, compute the CPU reference
// with ggml_graph_compute_with_ctx, export the graph via exporter.h, and write everything to a case
// directory that tests/test_mlir_export.py can compile and check without touching ggml.
//
// Links ggml only (never llama) - see the note in CMakeLists.txt.
//
//   mlir-export-cases --list
//   mlir-export-cases --emit <name> <dir>
//   mlir-export-cases --emit-all <dir>
#include "exporter.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

// mt19937 is standard-specified, so this is reproducible across stdlib implementations;
// std::uniform_real_distribution is not. Values land in [-1, 1].
static void fill_seeded(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    float *      d = (float *) t->data;
    const size_t n = ggml_nelements(t);
    for (size_t i = 0; i < n; i++) {
        d[i] = ((float) (rng() % 20001) - 10000.0f) / 10000.0f;
    }
}

// MLIR shape = ne reversed over n_dims (exporter.h mlir_shape_dims).
static std::vector<int64_t> mlir_shape_of(const ggml_tensor * t) {
    std::vector<int64_t> s;
    for (int i = ggml_n_dims(t) - 1; i >= 0; i--) {
        s.push_back(t->ne[i]);
    }
    return s;
}

static void write_f32(const fs::path & p, const ggml_tensor * t) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char *) t->data, (std::streamsize) (ggml_nelements(t) * sizeof(float)));
}

static std::string shape_json(const std::vector<int64_t> & s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); i++) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

// ---------------------------------------------------------------------------------------
// case definitions
// ---------------------------------------------------------------------------------------

// Builds the graph, appends every func-argument leaf to `args` in %arg order, returns the output.
using build_fn = ggml_tensor * (*) (ggml_context * ctx, std::vector<const ggml_tensor *> & args);

struct case_spec {
    const char * name;
    build_fn     build;
    float        rtol;
    float        atol;
    const char * expect;   // "pass" | "mismatch"
    bool         corrupt;  // deliberately poison expected_0.bin (harness self-check)
};

static ggml_tensor * build_add(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_add(ctx, a, b);
}

static ggml_tensor * build_mul(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul(ctx, a, b);
}

static ggml_tensor * build_scale(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_scale(ctx, a, 0.5f);   // scalar is baked into the graph, not a func arg
}

static ggml_tensor * build_silu(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_silu(ctx, a);          // GGML_OP_UNARY / GGML_UNARY_OP_SILU
}

// RMS_NORM normalizes over ne[0], so use 2-D input to exercise a real reduction per row.
static ggml_tensor * build_rms_norm(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_rms_norm(ctx, a, 1e-5f);
}

static ggml_tensor * build_soft_max(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 8);
    ggml_set_name(a, "a");
    args.push_back(a);
    return ggml_soft_max(ctx, a);
}

static const case_spec CASES[] = {
    { "add",          build_add, 0.0f, 0.0f, "pass",     false },
    // Proves the comparison in test_mlir_export.py actually compares. If a harness bug made the
    // check vacuous, every other case would still pass and this one would too - so this must fail
    // to match, by construction.
    { "add_negative", build_add, 0.0f, 0.0f, "mismatch", true  },
    { "mul",          build_mul,      0.0f,  0.0f,  "pass", false },
    { "scale",        build_scale,    0.0f,  0.0f,  "pass", false },
    { "silu",         build_silu,     1e-5f, 1e-6f, "pass", false },
    { "rms_norm",     build_rms_norm, 1e-5f, 1e-6f, "pass", false },
    { "soft_max",     build_soft_max, 1e-5f, 1e-6f, "pass", false },
};

static const size_t N_CASES = sizeof(CASES) / sizeof(CASES[0]);

// ---------------------------------------------------------------------------------------
// emit
// ---------------------------------------------------------------------------------------

static bool emit_case(const case_spec & spec, const fs::path & dir) {
    fs::create_directories(dir);

    ggml_init_params ip { (size_t) 256 << 20, nullptr, /*no_alloc=*/false };
    ggml_context *   ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", spec.name);
        return false;
    }

    std::vector<const ggml_tensor *> args;
    ggml_tensor *                    out = spec.build(ctx, args);

    // Seed per argument index, offset by a per-case hash so different cases get different data.
    uint32_t base = 0x9E3779B9u;
    for (const char * p = spec.name; *p; p++) base = base * 31u + (uint32_t) *p;
    for (size_t i = 0; i < args.size(); i++) {
        fill_seeded(const_cast<ggml_tensor *>(args[i]), base + (uint32_t) i);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_graph_compute_with_ctx failed\n", spec.name);
        ggml_free(ctx);
        return false;
    }

    std::string expect = spec.expect;
    std::string mlir;
    try {
        mlir = "module {\n" + build_func_text_baked(gf, "forward", args, {}) + "}\n";
    } catch (const mlir_export_error & e) {
        // Exporter gap: record it so the runner xfails with a reason instead of the build breaking.
        fprintf(stderr, "%s: exporter rejected the graph: %s\n", spec.name, e.what());
        expect = "unsupported";
        mlir   = "";
    }

    std::ofstream(dir / "forward.mlir") << mlir;

    std::string args_json;
    for (size_t i = 0; i < args.size(); i++) {
        std::string fn = "input_" + std::to_string(i) + ".bin";
        write_f32(dir / fn, args[i]);
        if (i) args_json += ",\n             ";
        args_json += "{\"file\": \"" + fn + "\", \"shape\": " + shape_json(mlir_shape_of(args[i])) + "}";
    }

    if (spec.corrupt) {
        // Offset element 0 by a large, unmistakable amount, then write.
        std::vector<float> ref(ggml_nelements(out));
        memcpy(ref.data(), out->data, ref.size() * sizeof(float));
        ref[0] += 1000.0f;
        std::ofstream f(dir / "expected_0.bin", std::ios::binary);
        f.write((const char *) ref.data(), (std::streamsize) (ref.size() * sizeof(float)));
    } else {
        write_f32(dir / "expected_0.bin", out);
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%.8g", spec.rtol);
    std::string rtol = buf;
    snprintf(buf, sizeof(buf), "%.8g", spec.atol);
    std::string atol = buf;

    std::ofstream(dir / "case.json")
        << "{\n"
        << "  \"name\": \"" << spec.name << "\",\n"
        << "  \"expect\": \"" << expect << "\",\n"
        << "  \"rtol\": " << rtol << ",\n"
        << "  \"atol\": " << atol << ",\n"
        << "  \"args\": [" << args_json << "],\n"
        << "  \"output\": {\"file\": \"expected_0.bin\", \"shape\": "
        << shape_json(mlir_shape_of(out)) << "}\n"
        << "}\n";

    ggml_free(ctx);
    printf("emitted %s -> %s\n", spec.name, dir.c_str());
    return true;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
        for (size_t i = 0; i < N_CASES; i++) printf("%s\n", CASES[i].name);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "--emit") == 0) {
        for (size_t i = 0; i < N_CASES; i++) {
            if (strcmp(CASES[i].name, argv[2]) == 0) {
                return emit_case(CASES[i], argv[3]) ? 0 : 1;
            }
        }
        fprintf(stderr, "unknown case: %s\n", argv[2]);
        return 1;
    }
    if (argc == 3 && strcmp(argv[1], "--emit-all") == 0) {
        for (size_t i = 0; i < N_CASES; i++) {
            if (!emit_case(CASES[i], fs::path(argv[2]) / CASES[i].name)) return 1;
        }
        return 0;
    }
    fprintf(stderr,
            "usage: %s --list\n"
            "       %s --emit <name> <dir>\n"
            "       %s --emit-all <dir>\n",
            argv[0], argv[0], argv[0]);
    return 1;
}
