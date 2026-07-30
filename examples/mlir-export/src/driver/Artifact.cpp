#include "Artifact.h"

#include "Config.h"

#include <dlfcn.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace tsi::driver {

namespace {

// FNV-1a over the module bytes. Not cryptographic, and does not need to be: it distinguishes IR that
// differs, and a collision would need two different graphs of the same model to hash alike.
// Memory-bandwidth bound, so ~0.1 s even for a 500 MiB module.
std::string hashOf(const std::string & bytes) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : bytes) {
        h = (h ^ (uint8_t) c) * 0x100000001b3ull;
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, h);
    return buf;
}

bool writeIfAbsent(const fs::path & path, const std::string & bytes) {
    std::error_code ec;
    if (fs::exists(path, ec) && fs::file_size(path, ec) == bytes.size()) {
        return true;   // same hash and same length: already written
    }
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), (std::streamsize) bytes.size());
    f.close();
    if (!f) {
        fprintf(stderr, "[tsi-mlir] cannot write %s\n", path.c_str());
        return false;
    }
    return true;
}

// Runs the compiler as a subprocess. In-process would mean embedding a Python interpreter and the
// tsavorite package inside llama, for no gain: the compile happens once per artifact.
bool compile(const fs::path & mlir, const fs::path & out, const Config & cfg) {
    if (!fs::exists(cfg.python)) {
        fprintf(stderr, "[tsi-mlir] no python at %s (set TSI_MLIR_PYTHON)\n", cfg.python.c_str());
        return false;
    }
    if (!fs::exists(cfg.script)) {
        fprintf(stderr, "[tsi-mlir] no compile script at %s (set TSI_MLIR_SCRIPT)\n",
                cfg.script.c_str());
        return false;
    }

    // TSI_RT_LIB_DIR is not optional: the script's final step links host.o against the runtime shim,
    // and without it the link fails on undefined tsi_* symbols after several minutes of compiling.
    // Passing our own value means the caller does not have to know that.
    std::string env;
    if (!cfg.rt_lib_dir.empty()) {
        env = "TSI_RT_LIB_DIR=\"" + cfg.rt_lib_dir + "\" ";
    }

    const std::string log = (out / "compile.log").string();
    const std::string cmd = env + "\"" + cfg.python + "\" \"" + cfg.script + "\" \"" +
                            mlir.string() + "\" \"" + out.string() + "\" > \"" + log + "\" 2>&1";
    fprintf(stderr, "[tsi-mlir] compiling (this takes minutes for a whole model) -> %s\n",
            out.c_str());
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        // The log holds the compiler's own diagnostics; pointing at it beats reprinting them here.
        fprintf(stderr, "[tsi-mlir] compile FAILED (rc=%d). See %s\n", rc, log.c_str());
        return false;
    }
    return true;
}

}  // namespace

forward_argv_fn buildForward(const std::string & mlir, const std::string & phase,
                             const Config & cfg) {
    const fs::path dir = fs::path(cfg.dir) / (phase + "-" + hashOf(mlir));
    const fs::path so  = dir / "host" / "host.so";

    std::error_code ec;
    fs::create_directories(dir, ec);

    const bool cached = fs::exists(so, ec);
    if (cached) {
        fprintf(stderr, "[tsi-mlir] %s: reusing cached %s\n", phase.c_str(), so.c_str());
    } else {
        const fs::path src = dir / "forward.mlirbc";
        if (!writeIfAbsent(src, mlir) || !compile(src, dir, cfg)) {
            return nullptr;
        }
        if (!fs::exists(so, ec)) {
            fprintf(stderr, "[tsi-mlir] %s: compile reported success but %s is missing\n",
                    phase.c_str(), so.c_str());
            return nullptr;
        }
    }

    // RTLD_LOCAL, not GLOBAL: prefill and decode each define @forward, at different arities.
    void * h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "[tsi-mlir] dlopen(%s) failed: %s\n", so.c_str(), dlerror());
        return nullptr;
    }
    auto fwd = (forward_argv_fn) dlsym(h, "tsi_forward_argv");
    if (!fwd) {
        fprintf(stderr, "[tsi-mlir] dlsym tsi_forward_argv failed: %s\n", dlerror());
    }
    return fwd;
}

}  // namespace tsi::driver
