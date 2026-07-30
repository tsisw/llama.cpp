#include "Config.h"

#include <cstdlib>

namespace tsi::driver {

namespace {

bool flag(const char * name) {
    const char * v = getenv(name);
    return v && v[0] && v[0] != '0';
}

std::string str(const char * name, const std::string & fallback) {
    const char * v = getenv(name);
    return (v && v[0]) ? std::string(v) : fallback;
}

// The compile script ships next to this source file, so derive its path from __FILE__ the way the
// tsavorite backend derives its blob paths. Moving or renaming the source tree breaks this, and
// TSI_MLIR_SCRIPT is the way out.
std::string defaultScript() {
    const std::string self = __FILE__;                    // .../examples/mlir-export/src/driver/Config.cpp
    const size_t      src  = self.rfind("/src/driver/");
    if (src == std::string::npos) {
        return "compile_graph_fpga.py";                   // built from somewhere unexpected; let PATH try
    }
    return self.substr(0, src) + "/compile_graph_fpga.py";
}

}  // namespace

const Config & Config::get() {
    static const Config cfg = [] {
        Config c;
        c.enabled = flag("TSI_MLIR_EXPORT");
        c.verify  = flag("TSI_MLIR_VERIFY");
        c.cpu_ref = flag("TSI_MLIR_CPU_REF");
        c.dump    = flag("TSI_MLIR_DUMP_GRAPH");
        c.weight_args = flag("TSI_MLIR_WEIGHT_ARGS");

        const char * skip = getenv("TSI_MLIR_SKIP");
        c.skip            = skip ? atoi(skip) : 1;

        c.dir    = str("TSI_MLIR_DIR", "./tsi-mlir");
        c.python = str("TSI_MLIR_PYTHON", std::string(getenv("HOME") ? getenv("HOME") : ".") +
                                              "/repo/mlir-compiler/venv/bin/python");
        c.script = str("TSI_MLIR_SCRIPT", defaultScript());
#ifdef TSI_RT_LIB_DIR_DEFAULT
        c.rt_lib_dir = str("TSI_RT_LIB_DIR", TSI_RT_LIB_DIR_DEFAULT);
#else
        c.rt_lib_dir = str("TSI_RT_LIB_DIR", "");
#endif
        return c;
    }();
    return cfg;
}

}  // namespace tsi::driver
