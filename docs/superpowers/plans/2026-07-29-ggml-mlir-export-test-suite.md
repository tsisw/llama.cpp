# ggml → MLIR Export Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A ctest-registered suite that takes small ggml graphs through export → compile → execute → numeric comparison against the TSI mlir-compiler, runnable on a macOS host build.

**Architecture:** Two stages joined by a case directory on disk. A C++ binary (`mlir-export-cases`) builds each ggml graph, computes the CPU reference, and writes `forward.mlir` + input/expected `.bin` + `case.json`. A pytest runner loads each case dir, JIT-compiles the MLIR via the mlir-compiler's `TXEBackend`, executes it, and compares. ctest runs the pytest stage, which generates cases itself via a session fixture so it is also standalone-runnable.

**Tech Stack:** C++20 (links `ggml` only), Python 3.11 from `~/repo/mlir-compiler/venv`, `tsavorite` + `tsi_mlir` packages, pytest, numpy, torch, CMake/ctest.

## Global Constraints

- Design doc: `docs/superpowers/specs/2026-07-29-ggml-mlir-export-test-suite-design.md`. Read it first.
- Host build only. `examples/mlir-export/` is already guarded by `if (NOT DEFINED GGML_TSAVORITE)` in `examples/CMakeLists.txt:34-37`. Do not remove that guard, and do not make any TSI-build target depend on this work.
- `mlir-export-cases` links **`ggml` only**, never `llama`. In a TSI build, linking `ggml` pulls in `libggml-tsavorite` and fails; `recon_cpu_check` sets the precedent.
- All CMake additions in `examples/mlir-export/CMakeLists.txt` must be inside `if (BUILD_TESTING)` for the `add_test` calls. `include(CTest)` runs at `CMakeLists.txt:303` only when `LLAMA_BUILD_TESTS=ON`; without the guard, configuring with `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON` errors.
- `MLIR_COMPILER_DIR` is a CMake cache PATH defaulting to `$ENV{HOME}/repo/mlir-compiler`. Missing venv → suite reports **skipped**, never a configure error.
- Execution target is a parameter. `ffm` is the default and the only one that works locally. `ten` requires Cadence `xt-clang` under `/proj/vendors/cadence/xtensa/…`, absent on macOS — it must skip with that reason, and must never be claimed to work.
- Float format on disk: raw `float32`, C order, native (little-endian) byte order. No headers.
- Determinism: fill inputs from `std::mt19937` with a per-case fixed seed, using integer modulo arithmetic (not `std::uniform_real_distribution`, which is not portable across standard libraries).
- CPU reference must use `ggml_graph_compute_with_ctx(ctx, gf, 1)` — one thread, for reproducibility.
- Build the C++ target with the working local toolchain: `~/repo/llama-cpp-build.sh` already configures clang-17 + `MacOSX15.4.sdk` + `GGML_METAL=OFF`.

## File Structure

| File | Responsibility |
|---|---|
| `examples/mlir-export/mlir_export_cases.cpp` (create) | Case definitions, seeded input fill, ggml CPU reference, exporter invocation, case-dir writer, CLI |
| `examples/mlir-export/tsi_raw_backend.py` (create) | `RawGraphBackend` — the one place that teaches `TXEBackend` to accept raw linalg MLIR text |
| `examples/mlir-export/compile_graph_fpga.py` (modify) | Import `RawGraphBackend` instead of defining it. CLI and behavior unchanged. |
| `examples/mlir-export/tests/conftest.py` (create) | pytest options (`--cases-bin`, `--cases-root`, `--target`), case-generation session fixture, case discovery/parameterization |
| `examples/mlir-export/tests/test_mlir_export.py` (create) | The single test body: load case, compile, run, compare |
| `examples/mlir-export/CMakeLists.txt` (modify) | `mlir-export-cases` target + `MLIR_COMPILER_DIR` + `add_test` |

Case generation lives in C++ because ggml graph construction is a C++ API. Compilation lives in Python because the compiler driver is a Python package. `conftest.py` holds all pytest plumbing so `test_mlir_export.py` stays a single readable assertion.

---

### Task 1: Extract `RawGraphBackend` into a shared module

`compile_graph_fpga.py` defines `RawGraphBackend` inline. The pytest runner needs the same class. Extract it so there is one definition.

**Files:**
- Create: `examples/mlir-export/tsi_raw_backend.py`
- Modify: `examples/mlir-export/compile_graph_fpga.py:22-39`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tsi_raw_backend.RawGraphBackend(TXEBackend)`, constructed as `RawGraphBackend(config)`, where `config` is a `tsavorite.compiler_config.TXECompilerConfig`. Its `.compile(model=<mlir text str>, input_types=[], compilation_type="jit"|"aot", output_dir=<str>, verbose=<bool>)` is inherited from `TXEBackend`.

- [ ] **Step 1: Write the failing test**

Create `examples/mlir-export/tests/test_tsi_raw_backend.py`:

```python
"""The extraction must leave both consumers importable and behaviorally identical."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_raw_backend_importable_and_overrides_convert_to_linalg():
    from tsi_raw_backend import RawGraphBackend
    from tsavorite.txe_backend.txe_backend import TXEBackend

    assert issubclass(RawGraphBackend, TXEBackend)
    # The whole point of the class: convert_to_linalg is overridden, not inherited.
    assert RawGraphBackend.convert_to_linalg is not TXEBackend.convert_to_linalg


def test_raw_backend_parses_linalg_text():
    from tsi_raw_backend import RawGraphBackend

    mlir = """
    module {
      func.func @forward(%arg0: tensor<4xf32> {txe.name = "input_0"})
          -> (tensor<4xf32> {txe.name = "res_0"}) attributes {llvm.emit_c_interface} {
        return %arg0 : tensor<4xf32>
      }
    }
    """
    mod = RawGraphBackend.convert_to_linalg(None, mlir, [])
    assert "func.func @forward" in str(mod)


def test_compile_graph_fpga_still_imports():
    # The AOT driver must keep working after the extraction.
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "compile_graph_fpga.py"
    spec = importlib.util.spec_from_file_location("compile_graph_fpga", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "RawGraphBackend")
    assert hasattr(mod, "main")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_tsi_raw_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsi_raw_backend'`

- [ ] **Step 3: Write the module**

Create `examples/mlir-export/tsi_raw_backend.py`:

```python
"""Teach the TSI compiler driver to accept an already-lowered linalg MLIR module.

TXEBackend subclasses normally implement convert_to_linalg to turn a torch/ONNX model into linalg.
The ggml exporter (exporter.h) already emits linalg MLIR text, so the "conversion" is just a parse.

Shared by compile_graph_fpga.py (AOT, for FPGA/posix bundles) and tests/test_mlir_export.py (JIT).
"""
from tsi_mlir.ir import Module
from tsavorite.txe_backend.txe_backend import TXEBackend


class RawGraphBackend(TXEBackend):
    """The model is already whole-graph linalg MLIR text; parse it into the txe context."""

    def convert_to_linalg(self, model, input_types, *, func_name=None, log_dir=None,
                          verbose=False, **kwargs):
        return Module.parse(model)
```

- [ ] **Step 4: Rewrite the import block in `compile_graph_fpga.py`**

Replace lines 22-39 (the `try: from tsi_mlir.ir import Module ... class RawGraphBackend ...` block) with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tsavorite.compiler_config import TXECompilerConfig
    from tsi_raw_backend import RawGraphBackend
except ImportError as e:
    print(
        f"error: {e}\nRun with a venv that has the mlir_external_packages wheel "
        f"(e.g. mlir-compiler/venv/bin/python3 {__file__})",
        file=sys.stderr,
    )
    sys.exit(1)
```

Note `import sys` and `from pathlib import Path` are already present at lines 18 and 20, and
`TXEBackend` / `Module` are no longer referenced by this file.

- [ ] **Step 5: Run tests to verify they pass**

Run: `~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_tsi_raw_backend.py -v`
Expected: 3 passed

- [ ] **Step 6: Verify the AOT CLI is not broken**

Run: `~/repo/mlir-compiler/venv/bin/python examples/mlir-export/compile_graph_fpga.py --help`
Expected: argparse usage text, exit 0. (Do not run a real compile; `--help` proves the import path.)

- [ ] **Step 7: Commit**

```bash
git add examples/mlir-export/tsi_raw_backend.py \
        examples/mlir-export/compile_graph_fpga.py \
        examples/mlir-export/tests/test_tsi_raw_backend.py
git commit -m "mlir-export: extract RawGraphBackend into a shared module"
```

---

### Task 2: C++ case generator with the `add` case

**Files:**
- Create: `examples/mlir-export/mlir_export_cases.cpp`
- Modify: `examples/mlir-export/CMakeLists.txt` (add the executable only; `add_test` comes in Task 3)

**Interfaces:**
- Consumes: `exporter.h`'s `build_func_text_baked(ggml_cgraph *, const char *, const std::vector<const ggml_tensor *> &, const std::vector<const ggml_tensor *> &) -> std::string` (returns the `func.func` block with **no** `module {}` wrapper — wrap it, as `decode_run.cpp:138` does) and `mlir_export_error`.
- Produces: binary `mlir-export-cases` with CLI `--list`, `--emit <name> <dir>`, `--emit-all <dir>`. Case dir contents: `forward.mlir`, `input_<i>.bin`, `expected_0.bin`, `case.json`. `case.json` schema:

```json
{
  "name": "add",
  "expect": "pass",
  "rtol": 0.0,
  "atol": 0.0,
  "args":   [{"file": "input_0.bin", "shape": [128]},
             {"file": "input_1.bin", "shape": [128]}],
  "output":  {"file": "expected_0.bin", "shape": [128]}
}
```

`shape` is the **MLIR** shape: ggml `ne` reversed over `ggml_n_dims(t)` dims, matching
`mlir_shape_dims` in `exporter.h:56-62`. A ggml 2-D tensor with `ne = [K, M]` has MLIR shape `[M, K]`.

- [ ] **Step 1: Write the failing test**

Create `examples/mlir-export/tests/test_cases_binary.py`:

```python
"""Stage 1 alone: the generator must produce a well-formed, parseable case dir."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES_BIN = Path(__file__).resolve().parents[3] / "build" / "bin" / "mlir-export-cases"


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_list_includes_add():
    out = subprocess.run([str(CASES_BIN), "--list"], capture_output=True, text=True, check=True)
    assert "add" in out.stdout.split()


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_emit_add_writes_wellformed_case(tmp_path):
    subprocess.run([str(CASES_BIN), "--emit", "add", str(tmp_path)], check=True)

    meta = json.loads((tmp_path / "case.json").read_text())
    assert meta["name"] == "add"
    assert meta["expect"] == "pass"
    assert len(meta["args"]) == 2

    mlir = (tmp_path / "forward.mlir").read_text()
    assert mlir.lstrip().startswith("module {")
    assert "func.func @forward" in mlir
    assert "llvm.emit_c_interface" in mlir

    # Reference must be the real elementwise sum of the two inputs we wrote out.
    a = np.fromfile(tmp_path / "input_0.bin", dtype=np.float32)
    b = np.fromfile(tmp_path / "input_1.bin", dtype=np.float32)
    exp = np.fromfile(tmp_path / "expected_0.bin", dtype=np.float32)
    n = int(np.prod(meta["output"]["shape"]))
    assert exp.size == n
    np.testing.assert_array_equal(exp, a + b)


@pytest.mark.skipif(not CASES_BIN.exists(), reason=f"{CASES_BIN} not built")
def test_emit_is_deterministic(tmp_path):
    d1, d2 = tmp_path / "r1", tmp_path / "r2"
    for d in (d1, d2):
        d.mkdir()
        subprocess.run([str(CASES_BIN), "--emit", "add", str(d)], check=True)
    for f in ("input_0.bin", "input_1.bin", "expected_0.bin", "forward.mlir"):
        assert (d1 / f).read_bytes() == (d2 / f).read_bytes(), f"{f} not deterministic"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_cases_binary.py -v`
Expected: 3 skipped (binary not built). That skip *is* the failing state — the binary does not exist.

- [ ] **Step 3: Write the case generator**

Create `examples/mlir-export/mlir_export_cases.cpp`:

```cpp
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

static const case_spec CASES[] = {
    { "add", build_add, 0.0f, 0.0f, "pass" },
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
        args_json += "{\"file\": \"" + fn + "\", \"shape\" : " + shape_json(mlir_shape_of(args[i])) + "}";
    }

    write_f32(dir / "expected_0.bin", out);

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
```

- [ ] **Step 4: Add the executable to CMake**

Append to `examples/mlir-export/CMakeLists.txt`:

```cmake
# Case generator for the export test suite. Links ggml ONLY (not llama), same reason as the
# checks above: in a TSI build linking llama/ggml pulls in libggml-tsavorite and the link fails.
add_executable(mlir-export-cases mlir_export_cases.cpp)
target_link_libraries(mlir-export-cases PRIVATE ggml)
target_compile_features(mlir-export-cases PRIVATE cxx_std_20)
```

- [ ] **Step 5: Build it**

Run: `cmake --build ~/repo/llama.cpp/build --target mlir-export-cases`
Expected: links, produces `build/bin/mlir-export-cases`

- [ ] **Step 6: Run tests to verify they pass**

Run: `~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_cases_binary.py -v`
Expected: 3 passed

- [ ] **Step 7: Eyeball the emitted MLIR once**

Run: `./build/bin/mlir-export-cases --emit add /tmp/case-add && cat /tmp/case-add/forward.mlir`
Expected: a `module { func.func @forward(%arg0: tensor<128xf32> …, %arg1: tensor<128xf32> …) -> (tensor<128xf32> …)` with a `linalg.generic` or equivalent add body. If the arg types are not `tensor<128xf32>`, stop — the shape convention is wrong and every later case inherits the bug.

- [ ] **Step 8: Commit**

```bash
git add examples/mlir-export/mlir_export_cases.cpp \
        examples/mlir-export/CMakeLists.txt \
        examples/mlir-export/tests/test_cases_binary.py
git commit -m "mlir-export: add case generator with the add case"
```

---

### Task 3: pytest runner (FFM) + ctest registration

First true end-to-end: export → compile → execute → compare.

**Files:**
- Create: `examples/mlir-export/tests/conftest.py`
- Create: `examples/mlir-export/tests/test_mlir_export.py`
- Modify: `examples/mlir-export/CMakeLists.txt`

**Interfaces:**
- Consumes: `tsi_raw_backend.RawGraphBackend` (Task 1); `mlir-export-cases --emit-all <dir>` (Task 2); the `case.json` schema (Task 2).
- Produces: pytest options `--cases-bin <path>`, `--cases-root <path>`, `--target ffm|ten`; fixture `case_dir` parameterized over discovered cases; ctest test named `mlir-export-suite` with label `mlir-export`.

- [ ] **Step 1: Write conftest.py**

Create `examples/mlir-export/tests/conftest.py`:

```python
"""pytest plumbing for the ggml->MLIR export suite.

Cases come from the C++ generator. Either point --cases-root at a directory that already holds
case dirs (stage 2 in isolation), or pass --cases-bin and let the session fixture generate them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption("--cases-bin", default=None,
                     help="path to the mlir-export-cases binary; generates cases into a tmp dir")
    parser.addoption("--cases-root", default=None,
                     help="directory of pre-generated case dirs (skips generation)")
    parser.addoption("--target", default="ffm", choices=["ffm", "ten"],
                     help="ffm = host-native JIT (default); ten = TXE sim, needs Cadence xt-clang")


XT_CLANG = Path("/proj/vendors/cadence/xtensa/XtDevTools/install/tools/"
                "RJ-2025.5-linux/XtensaTools/bin/xt-clang")


@pytest.fixture(scope="session")
def target(request):
    t = request.config.getoption("--target")
    if t == "ten" and not XT_CLANG.exists():
        pytest.skip(f"--target ten needs the Cadence Xtensa toolchain ({XT_CLANG}); "
                    f"not present on this machine, so TXE blobs cannot be built")
    return t


@pytest.fixture(scope="session")
def cases_root(request, tmp_path_factory):
    root = request.config.getoption("--cases-root")
    if root:
        return Path(root)
    bin_path = request.config.getoption("--cases-bin")
    if not bin_path or not Path(bin_path).exists():
        pytest.skip("pass --cases-bin <mlir-export-cases> or --cases-root <dir>")
    out = tmp_path_factory.mktemp("mlir-export-cases")
    subprocess.run([str(bin_path), "--emit-all", str(out)], check=True)
    return out


def _discover(config):
    """Case names, resolved at collection time so each case is its own test id."""
    root = config.getoption("--cases-root")
    if root:
        return sorted(p.parent.name for p in Path(root).glob("*/case.json"))
    bin_path = config.getoption("--cases-bin")
    if bin_path and Path(bin_path).exists():
        out = subprocess.run([str(bin_path), "--list"], capture_output=True, text=True, check=True)
        return sorted(out.stdout.split())
    return []


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        names = _discover(metafunc.config)
        metafunc.parametrize("case_name", names or [pytest.param("<none>", marks=pytest.mark.skip(
            reason="no cases discovered; pass --cases-bin or --cases-root"))])


@pytest.fixture
def case(case_name, cases_root):
    d = cases_root / case_name
    meta = json.loads((d / "case.json").read_text())
    return d, meta
```

- [ ] **Step 2: Write the test body**

Create `examples/mlir-export/tests/test_mlir_export.py`:

```python
"""End-to-end: a ggml graph exported to linalg MLIR compiles and executes with numerics matching
ggml's own CPU result.

Each case directory is produced by mlir-export-cases. case.json's "expect" field decides the
assertion: pass -> must match, unsupported -> xfail with the exporter's reason, mismatch -> must
NOT match (that case exists to prove this comparison is not vacuous).
"""
import numpy as np
import pytest
import torch

from tsavorite.compiler_config import TXECompilerConfig
from tsi_raw_backend import RawGraphBackend


def _config(target):
    if target == "ten":
        return TXECompilerConfig.for_ten(log_mlir=True, enable_tvu=True, enable_tmu=True)
    return TXECompilerConfig(log_mlir=True)


def _load(path, shape):
    a = np.fromfile(path, dtype=np.float32)
    return a.reshape(shape)


def test_case_matches_ggml_reference(case, target, tmp_path):
    case_dir, meta = case

    if meta["expect"] == "unsupported":
        pytest.xfail(f"exporter does not support case {meta['name']!r} "
                     f"(see stderr from mlir-export-cases)")

    mlir = (case_dir / "forward.mlir").read_text()
    assert "func.func @forward" in mlir, "case emitted no forward function"

    inputs = [torch.from_numpy(_load(case_dir / a["file"], a["shape"]).copy())
              for a in meta["args"]]
    expected = _load(case_dir / meta["output"]["file"], meta["output"]["shape"])

    runner = RawGraphBackend(_config(target)).compile(
        model=mlir, input_types=[], compilation_type="jit",
        output_dir=str(tmp_path / "compiled"), verbose=False,
    )

    runner.runtime_init()
    try:
        got = runner.forward(*inputs)
    finally:
        runner.runtime_finalize()

    got = np.asarray(got.detach() if hasattr(got, "detach") else got, dtype=np.float32)
    got = got.reshape(expected.shape)

    close = np.allclose(got, expected, rtol=meta["rtol"], atol=meta["atol"])

    if meta["expect"] == "mismatch":
        assert not close, (
            f"case {meta['name']!r} carries a deliberately corrupted reference but compared EQUAL. "
            f"The comparison is not actually checking anything."
        )
        return

    if not close:
        diff = np.abs(got - expected)
        idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
        denom = np.maximum(np.abs(expected), 1e-30)
        raise AssertionError(
            f"case {meta['name']!r} mismatch: max abs err {diff.max():.3e} at {idx} "
            f"(got {got[idx]:.8g} vs expected {expected[idx]:.8g}), "
            f"max rel err {np.max(diff / denom):.3e}, "
            f"tolerance rtol={meta['rtol']} atol={meta['atol']}"
        )
```

- [ ] **Step 3: Run it to verify the `add` case passes end to end**

Run:
```bash
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-bin ./build/bin/mlir-export-cases -v
```
Expected: `test_case_matches_ggml_reference[add] PASSED`. If it fails at compile, read `tmp_path/compiled/*.mlir` (`log_mlir=True` keeps every stage) to find which stage rejected the module.

- [ ] **Step 4: Register with ctest**

Append to `examples/mlir-export/CMakeLists.txt`:

```cmake
# ctest wiring. BUILD_TESTING is set by include(CTest), which the top-level CMakeLists.txt runs at
# line 303 only when LLAMA_BUILD_TESTS=ON - without this guard, configuring with
# -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON would fail on add_test().
if (BUILD_TESTING)
    set(MLIR_COMPILER_DIR "$ENV{HOME}/repo/mlir-compiler" CACHE PATH
        "TSI mlir-compiler checkout; must contain venv/bin/python with the tsavorite package")
    set(TSI_VENV_PYTHON "${MLIR_COMPILER_DIR}/venv/bin/python")

    if (EXISTS "${TSI_VENV_PYTHON}")
        add_test(NAME mlir-export-suite
                 COMMAND "${TSI_VENV_PYTHON}" -m pytest -v
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mlir_export.py"
                         --cases-bin "$<TARGET_FILE:mlir-export-cases>"
                         --target ffm)
        set_tests_properties(mlir-export-suite PROPERTIES LABELS "mlir-export" TIMEOUT 1800)
    else()
        # Never a hard error: a plain host build must not start requiring the compiler repo.
        message(STATUS "mlir-export: ${TSI_VENV_PYTHON} not found; mlir-export-suite will skip "
                       "(configure with -DMLIR_COMPILER_DIR=<path> to enable)")
        add_test(NAME mlir-export-suite COMMAND ${CMAKE_COMMAND} -E echo
                 "SKIPPED: mlir-compiler venv not found at ${TSI_VENV_PYTHON}")
        set_tests_properties(mlir-export-suite PROPERTIES LABELS "mlir-export"
                             SKIP_REGULAR_EXPRESSION "SKIPPED")
    endif()
endif()
```

- [ ] **Step 5: Reconfigure and run through ctest**

Run:
```bash
cmake -B build -DGGML_METAL=OFF
ctest --test-dir build -R mlir-export-suite --output-on-failure
```
Expected: 1 test, passed.

- [ ] **Step 6: Verify the skip path is real, not theoretical**

Run:
```bash
cmake -B /tmp/skipcheck -G Ninja -DGGML_METAL=OFF -DGGML_TSAVORITE_TARGET=posix \
  -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@17/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@17/bin/clang++ \
  -DMLIR_COMPILER_DIR=/nonexistent 2>&1 | grep "mlir-export:"
```
Expected: the `not found; mlir-export-suite will skip` STATUS line, and configure exits 0.
Then `rm -rf /tmp/skipcheck`.

- [ ] **Step 7: Commit**

```bash
git add examples/mlir-export/tests/conftest.py \
        examples/mlir-export/tests/test_mlir_export.py \
        examples/mlir-export/CMakeLists.txt
git commit -m "mlir-export: add pytest runner and ctest registration"
```

---

### Task 4: The vacuous-suite guard

The most important test here. Everything above could be green while comparing nothing.

**Files:**
- Modify: `examples/mlir-export/mlir_export_cases.cpp`

**Interfaces:**
- Consumes: `case_spec`, `emit_case` (Task 2); the `expect: "mismatch"` branch in `test_mlir_export.py` (Task 3).
- Produces: case `add_negative`.

- [ ] **Step 1: Add the corrupting case**

In `mlir_export_cases.cpp`, add a `corrupt` flag to `case_spec` (default `false`), set it on the new
entry, and after the `write_f32(dir / "expected_0.bin", out);` line, perturb the reference when set.

Change the struct to:

```cpp
struct case_spec {
    const char * name;
    build_fn     build;
    float        rtol;
    float        atol;
    const char * expect;   // "pass" | "mismatch"
    bool         corrupt;  // deliberately poison expected_0.bin (harness self-check)
};
```

Update the existing entry and add the new one:

```cpp
static const case_spec CASES[] = {
    { "add",          build_add, 0.0f, 0.0f, "pass",     false },
    // Proves the comparison in test_mlir_export.py actually compares. If a harness bug made the
    // check vacuous, every other case would still pass and this one would too - so this must fail
    // to match, by construction.
    { "add_negative", build_add, 0.0f, 0.0f, "mismatch", true  },
};
```

Replace the single reference write with:

```cpp
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
```

- [ ] **Step 2: Rebuild**

Run: `cmake --build build --target mlir-export-cases`
Expected: builds clean.

- [ ] **Step 3: Confirm the guard behaves as designed**

Run:
```bash
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-bin ./build/bin/mlir-export-cases -v -k "add_negative or add]"
```
Expected: both `[add]` and `[add_negative]` PASS — `add_negative` passes *because* it correctly
detects the mismatch.

- [ ] **Step 4: Prove the guard can actually fail (invert it once)**

Temporarily change `ref[0] += 1000.0f;` to `ref[0] += 0.0f;`, rebuild, rerun the `add_negative` test.
Expected: it FAILS with "carries a deliberately corrupted reference but compared EQUAL". Then revert
the line to `+= 1000.0f`, rebuild, and confirm it passes again. A guard never observed failing is
not a guard.

- [ ] **Step 5: Commit**

```bash
git add examples/mlir-export/mlir_export_cases.cpp
git commit -m "mlir-export: add add_negative harness self-check"
```

---

### Task 5: Elementwise, unary and reduction cases

**Files:**
- Modify: `examples/mlir-export/mlir_export_cases.cpp`

**Interfaces:**
- Consumes: `case_spec`, `build_fn`, `emit_case` (Tasks 2, 4).
- Produces: cases `mul`, `scale`, `silu`, `rms_norm`, `soft_max`.

- [ ] **Step 1: Add the builders**

Insert after `build_add`:

```cpp
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
```

- [ ] **Step 2: Register them**

Extend `CASES` (keep `add` and `add_negative`):

```cpp
static const case_spec CASES[] = {
    { "add",          build_add,      0.0f,  0.0f,  "pass",     false },
    { "add_negative", build_add,      0.0f,  0.0f,  "mismatch", true  },
    { "mul",          build_mul,      0.0f,  0.0f,  "pass",     false },
    { "scale",        build_scale,    0.0f,  0.0f,  "pass",     false },
    { "silu",         build_silu,     1e-5f, 1e-6f, "pass",     false },
    { "rms_norm",     build_rms_norm, 1e-5f, 1e-6f, "pass",     false },
    { "soft_max",     build_soft_max, 1e-5f, 1e-6f, "pass",     false },
};
```

- [ ] **Step 3: Rebuild and run**

Run:
```bash
cmake --build build --target mlir-export-cases
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-bin ./build/bin/mlir-export-cases -v
```
Expected: 7 passed, or an xfail for any op the exporter rejects (that is a legitimate finding, not a
failure to fix here).

- [ ] **Step 4: Calibrate tolerances from observed error, deliberately**

If a case fails only on tolerance, the assertion message prints the observed max abs and max rel
error. Raise that case's `rtol`/`atol` to just above the observed value and add a trailing comment
recording the measured number, e.g. `1e-5f, /* observed 5.7e-06 */ 1e-5f`. Do **not** widen a
tolerance to hide a real numeric divergence — a rel error above ~1e-4 on f32 is a bug, not noise.

- [ ] **Step 5: Commit**

```bash
git add examples/mlir-export/mlir_export_cases.cpp
git commit -m "mlir-export: add mul, scale, silu, rms_norm, soft_max cases"
```

---

### Task 6: matmul and composite cases

**Files:**
- Modify: `examples/mlir-export/mlir_export_cases.cpp`

**Interfaces:**
- Consumes: `case_spec`, `build_fn`, `emit_case` (Tasks 2, 4, 5).
- Produces: cases `matmul`, `matmul_add`.

`attn_small` is deferred by request (2026-07-29) — see the spec's Out of scope. `matmul_add` is the
composite that covers op composition.

Shape note, verified against `exporter.h:284` (`emit_mul_mat_2d`): `ggml_mul_mat(ctx, a, b)` requires
`a->ne[0] == b->ne[0]` (= K) and yields `ne = (a->ne[1], b->ne[1])`. In MLIR shape order that is
`a -> [M,K]`, `b -> [N,K]`, `out -> [N,M]`, computed as `B @ Aᵀ`. K is kept a multiple of 32 for the
TMU K-alignment (`TMU_K_MULTIPLE` in `ggml/include/ggml-tsavorite.h`).

A probe on 2026-07-29 confirmed this shape lowers and runs correctly on the **default** FFM config at
M=K=N=32 (rel l2 1.2e-07, max abs 5.7e-06) and that a `tmu_mma_shape=[8,1]` override changes nothing.
So no config override is needed here, unlike the FPGA path in `compile_graph_fpga.py:71`.

- [ ] **Step 1: Add the builders**

```cpp
static ggml_tensor * build_matmul(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int K = 32, M = 32, N = 32;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);   // MLIR [M,K]
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);   // MLIR [N,K]
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    args.push_back(a);
    args.push_back(b);
    return ggml_mul_mat(ctx, a, b);                                   // ne (M,N) -> MLIR [N,M]
}

static ggml_tensor * build_matmul_add(ggml_context * ctx, std::vector<const ggml_tensor *> & args) {
    const int K = 32, M = 32, N = 32;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, M, N);   // matches mul_mat's ne
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");
    args.push_back(a);
    args.push_back(b);
    args.push_back(c);
    return ggml_add(ctx, ggml_mul_mat(ctx, a, b), c);
}
```

- [ ] **Step 2: Register them**

```cpp
    { "matmul",     build_matmul,     1e-5f, 1e-5f, "pass", false },
    { "matmul_add", build_matmul_add, 1e-5f, 1e-5f, "pass", false },
```

`atol` is 1e-5 here, not the 1e-6 used for the elementwise cases, because the probe measured
max abs err 5.7e-06 on a 32×32×32 f32 matmul — reduction reassociation, not a defect.

- [ ] **Step 3: Rebuild and run the full suite**

Run:
```bash
cmake --build build --target mlir-export-cases
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-bin ./build/bin/mlir-export-cases -v
```
Expected: 9 tests. `matmul` should pass on the strength of the probe. If `matmul_add` xfails on an
exporter gap, record the exporter message in the commit body and leave it xfailing rather than
changing the exporter (out of scope per the spec).

- [ ] **Step 4: Record the wall-clock cost**

Run: `time ctest --test-dir build -R mlir-export-suite`
If total runtime exceeds ~2 minutes, split `matmul` and `matmul_add` into a second ctest test
labelled `mlir-export-slow` so the fast suite stays usable, and note the split in the spec's Risks
table.

- [ ] **Step 5: Commit**

```bash
git add examples/mlir-export/mlir_export_cases.cpp
git commit -m "mlir-export: add matmul and matmul_add cases"
```

---

### Task 7: `--target ten` verification and documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `~/repo/llama-cpp-build.sh`

**Interfaces:**
- Consumes: the `target` fixture and `XT_CLANG` skip (Task 3); ctest test `mlir-export-suite` (Task 3).
- Produces: no code interfaces. Documentation and a build-script hook.

- [ ] **Step 1: Verify the `ten` skip fires with its real reason**

Run:
```bash
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-bin ./build/bin/mlir-export-cases --target ten -v
```
Expected: all cases SKIPPED, reason naming the missing Cadence Xtensa toolchain. Not an error, and
not a pass — a skip. If anything reports passing under `--target ten` on macOS, the skip guard is
wrong: blobs cannot be built without `xt-clang`.

- [ ] **Step 2: Document the suite in CLAUDE.md**

Add to the "Running and testing" section of `CLAUDE.md`:

```markdown
### ggml → MLIR export suite

Verifies the exporter end to end: ggml graph → linalg MLIR → TSI compiler → JIT execute → compare
against ggml's own CPU result. Host builds only; needs the TSI compiler checkout.

```sh
ctest --test-dir build -R mlir-export-suite --output-on-failure
```

Configure with `-DMLIR_COMPILER_DIR=<path>` if the compiler is not at `~/repo/mlir-compiler`; if the
venv is missing the suite reports skipped rather than failing. Cases are generated by
`build/bin/mlir-export-cases` (`--list`, `--emit <name> <dir>`, `--emit-all <dir>`), so either stage
can be run alone when debugging:

```sh
./build/bin/mlir-export-cases --emit matmul /tmp/c
~/repo/mlir-compiler/venv/bin/python -m pytest examples/mlir-export/tests/test_mlir_export.py \
    --cases-root /tmp --target ffm -v -k matmul
```

`--target ffm` (default) runs the host-native functional model. `--target ten` targets the TXE
simulator and **only works on an SDK box** — it needs Cadence `xt-clang` to build TXE blobs and
skips everywhere else. `add_negative` is a deliberate-mismatch case that must pass by *detecting*
a corrupted reference; if it ever fails, the comparison logic is broken.
```

- [ ] **Step 3: Add a suite hook to the build script**

In `~/repo/llama-cpp-build.sh`, inside the `if [ "$DO_TEST" -eq 1 ]; then` block, after the
`test-backend-ops` invocation:

```bash
    # Export suite: skips cleanly if the mlir-compiler venv is absent.
    echo "--- mlir-export suite"
    ctest --test-dir "$BUILD_DIR" -R mlir-export-suite --output-on-failure || true
```

Also add `MLIR_COMPILER_DIR` to the script's env-override list in the header comment, defaulting to
`$HOME/repo/mlir-compiler`, and pass it through to the configure step as
`-DMLIR_COMPILER_DIR="$MLIR_COMPILER_DIR"`.

- [ ] **Step 4: Full verification run**

Run:
```bash
~/repo/llama-cpp-build.sh --test 2>&1 | tail -30
```
Expected: build succeeds; ctest shows the upstream 39/40 (the `test-tokenizers-ggml-vocabs` git-lfs
failure is pre-existing and unrelated), `test-backend-ops` OK, and `mlir-export-suite` passing.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the ggml->MLIR export test suite"
```

(`~/repo/llama-cpp-build.sh` lives outside the repo and is not committed.)

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: two-stage architecture → Tasks 2+3; case-dir
format → Task 2 Interfaces; `mlir_export_cases.cpp` → Task 2; `tsi_raw_backend.py` extraction →
Task 1; pytest runner → Task 3; CMake/ctest + `MLIR_COMPILER_DIR` + skip-not-fail → Task 3 Steps 4-6;
the eight `pass` cases → Tasks 2, 5, 6; `add_negative` guard → Task 4; error handling (xfail on
unsupported, stage-named compile failure, error-detail on mismatch, venv skip, ten skip) → Task 3
Step 2 and Task 7 Step 1; out-of-scope items are not implemented anywhere, as intended.

**Placeholder scan.** No TBD/TODO. Every code step carries complete code. Task 5 Step 4 and Task 6
Step 4 are conditional but state an explicit decision rule and threshold rather than "adjust as
needed".

**Type consistency.** `case_spec` gains its `corrupt` field in Task 4 and is re-declared in full
there; Tasks 5 and 6 use the 6-field initializer form consistently. `build_fn` signature is identical
across all seven builders. `case.json` keys (`name`, `expect`, `rtol`, `atol`, `args[].file`,
`args[].shape`, `output.file`, `output.shape`) are written in Task 2 and read with the same names in
Task 3. Fixture names (`case_name`, `cases_root`, `case`, `target`) match between `conftest.py` and
`test_mlir_export.py`. `RawGraphBackend(config).compile(model=, input_types=, compilation_type=,
output_dir=, verbose=)` is used identically in Task 1's module docstring, Task 3's runner, and the
existing `compile_graph_fpga.py`.

One known ordering constraint: Task 4 edits the `CASES` array that Task 5 and 6 extend, so those
three tasks must be done in order. Tasks 1 and 2 are independent of each other.
