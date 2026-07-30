# Whole-graph ggml -> linalg MLIR exporter, its host-side CPU checks, and the export test suite.
#
# Build with a non-TSI host cmake config, not the cross-compiled fpga package build (there, linking
# ggml pulls in libggml-tsavorite and the link fails):
#
#   cmake -B build-host -DLLAMA_CURL=OFF -DGGML_TSAVORITE_TARGET=posix
#   cmake --build build-host --target ref_check recon_cpu_check decode_cpu_check test-mlir-export-cases
#
# ref_check prints llama.cpp's prompt ids and greedy next token; recon_cpu_check rebuilds build_layer
# on CPU for the same ids; decode_cpu_check checks the KV-cache decode step reproduces the prefill.
#   ./build-host/bin/ref_check          model.gguf "hello world"     -> ids + reference token
#   ./build-host/bin/recon_cpu_check    model.gguf <id0> <id1> ...   -> recon argmax per column
#   ./build-host/bin/decode_cpu_check   model.gguf <id0> <id1> ...   -> decode-vs-prefill MATCH per step
#
# Layout: include/tsi/{export,graph} public headers, src/{dialect,import,convert,export} the exporter
# library, tools/<name> one executable each, tests/ the export test suite (its C++ case generator
# included). compile_graph_fpga.py and tsi_raw_backend.py stay at this directory's top level because
# tsi-pkg-build.sh copies them from here by name, and the former imports the latter as a sibling.

set(TSI_MLIR_EXPORT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

set(MLIR_COMPILER_DIR "$ENV{HOME}/repo/mlir-compiler" CACHE PATH
    "TSI mlir-compiler checkout; supplies both the MLIR C++ libraries and the venv used by the tests")

# ---------------------------------------------------------------------------------------------
# MLIR C++ libraries, from mlir-compiler's own LLVM (22.0.0git) - never a system or Homebrew LLVM,
# which would be a different version and would not match the compiler consuming our output.
#
# Two constraints, both verified rather than assumed:
#   * The enclosing project MUST enable C. find_package(MLIR) -> LLVMConfig -> FindLibEdit runs a C
#     check_include_file, which errors out in a CXX-only project. llama.cpp's top-level
#     project("llama.cpp" C CXX) satisfies this.
#   * Do NOT include(HandleLLVMOptions). It would add -fno-exceptions (their LLVM is built with
#     LLVM_ENABLE_EH=OFF) and break `throw mlir_export_error`. Include dirs plus link libs is all
#     that is needed; exceptions then work fine against those static libs.
# ---------------------------------------------------------------------------------------------
if (NOT MLIR_DIR)
    # A build tree, not an install tree. Also accept an install prefix layout.
    foreach (candidate "${MLIR_COMPILER_DIR}/build/_deps/llvm-build/lib/cmake/mlir"
                       "${MLIR_COMPILER_DIR}/install/lib/cmake/mlir")
        if (EXISTS "${candidate}/MLIRConfig.cmake")
            set(MLIR_DIR "${candidate}")
            break()
        endif()
    endforeach()
endif()

# NO_DEFAULT_PATH is load-bearing, not defensive tidiness. A plain find_package(MLIR CONFIG) ignores
# an MLIR_DIR that does not resolve and falls back to CMake's default search paths, which on a machine
# with Homebrew LLVM installed silently finds MLIR 17 and builds the exporter against a different
# MLIR version than the compiler that consumes its output. Only ever look where we were pointed.
if (MLIR_DIR AND EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
    find_package(MLIR REQUIRED CONFIG NO_DEFAULT_PATH PATHS "${MLIR_DIR}")
endif()

if (MLIR_FOUND)
    message(STATUS "mlir-export: MLIR ${LLVM_PACKAGE_VERSION} from ${MLIR_DIR}")

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    # tablegen() derives its -I flags from DIRECTORY-scope include_directories, not from target
    # properties. Setting these only per-target makes .td files fail with
    # "could not find include file 'mlir/IR/OpBase.td'".
    include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

    # --- the ggml dialect ---------------------------------------------------------------------
    set(LLVM_TARGET_DEFINITIONS src/dialect/GgmlOps.td)
    mlir_tablegen(GgmlOps.h.inc -gen-op-decls)
    mlir_tablegen(GgmlOps.cpp.inc -gen-op-defs)
    mlir_tablegen(GgmlOpsDialect.h.inc -gen-dialect-decls -dialect=ggml)
    mlir_tablegen(GgmlOpsDialect.cpp.inc -gen-dialect-defs -dialect=ggml)
    add_public_tablegen_target(TsiGgmlOpsIncGen)

    # Only the dialects the exporter builds, not all 412 static libs.
    add_library(tsi-mlir-export STATIC
        src/dialect/GgmlDialect.cpp
        src/import/Importer.cpp
        src/convert/GgmlToLinalg.cpp
        src/convert/PatternSupport.cpp
        src/convert/PatternsElementwise.cpp
        src/convert/PatternsNorm.cpp
        src/convert/PatternsMatmul.cpp
        src/convert/PatternsShape.cpp
        src/convert/PatternsRope.cpp
        src/export/Builder.cpp
        src/export/Exporter.cpp
        src/export/GraphWalk.cpp
    )
    add_dependencies(tsi-mlir-export TsiGgmlOpsIncGen)
    target_include_directories(tsi-mlir-export
        PUBLIC  "${TSI_MLIR_EXPORT_INCLUDE}"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/dialect"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/export"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/import"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/convert"
                "${CMAKE_CURRENT_BINARY_DIR}"          # generated .inc files land here
                ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    # memref + bufferization are for the KV cache path only (CacheSpec in Exporter.h).
    target_link_libraries(tsi-mlir-export PRIVATE
        ggml MLIRIR MLIRParser MLIRFuncDialect MLIRLinalgDialect MLIRTensorDialect
        MLIRArithDialect MLIRMathDialect MLIRSupport
        MLIRMemRefDialect MLIRBufferizationDialect
        MLIRBytecodeWriter                        # Format::Bytecode
        MLIRPass MLIRTransforms MLIRTransformUtils)
    # C++17 to match the standard their LLVM was built with. Consumers stay at C++20; they only see
    # the MLIR-free public header.
    target_compile_features(tsi-mlir-export PRIVATE cxx_std_17)

    # ---------------------------------------------------------------------------------------------
    # Whole-graph interception hooked straight into llama_context, so `llama-cli` alone does
    # capture/compile/run with no wrapper scripts.
    #
    # This needs the TSI runtime shim, because the compiled host.o is an ORCHESTRATION program: it
    # calls tsi_initialize/tsi_alloc/tsi_copy/tsi_load_blob/tsi_launch_blob/tsi_wait rather than
    # computing anything itself. That is true for the FFM target too - FFM only changes where the
    # kernels live (host txe_blob_*.o instead of TXE .blob), not the host API.
    #
    # The shim is built by mlir-compiler's setup_runtime() into _deps/runtime-build, NOT build/lib.
    # ---------------------------------------------------------------------------------------------
    set(TSI_RT_INCLUDE_DIR "${MLIR_COMPILER_DIR}/third-party/runtime/include/tsi-rt/shim")
    set(TSI_RT_LIB_DIR     "${MLIR_COMPILER_DIR}/build/_deps/runtime-build/lib")
    find_library(TSI_RT_SHIM_LIB NAMES TsavRTShimCAPI PATHS "${TSI_RT_LIB_DIR}" NO_DEFAULT_PATH)

    if (TSI_RT_SHIM_LIB AND EXISTS "${TSI_RT_INCLUDE_DIR}/HostShimCAPI.h" AND TARGET llama)
        message(STATUS "mlir-export: TSI runtime shim ${TSI_RT_SHIM_LIB}")
        add_library(tsi-llama-hook STATIC src/llama/WholeGraphHook.cpp)
        add_dependencies(tsi-llama-hook TsiGgmlOpsIncGen)
        target_include_directories(tsi-llama-hook PRIVATE
            "${TSI_MLIR_EXPORT_INCLUDE}"
            "${TSI_RT_INCLUDE_DIR}"                                  # bare "HostShimCAPI.h"
            "${MLIR_COMPILER_DIR}/third-party/runtime/include"       # its own "tsi-rt/..." includes
            "${MLIR_COMPILER_DIR}/build/_deps/runtime-build/include" # generated runtime headers
            "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-tsavorite/include"   # TestModel.h
            "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src"
            ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
        target_link_libraries(tsi-llama-hook PUBLIC tsi-mlir-export ggml "${TSI_RT_SHIM_LIB}")
        target_link_directories(tsi-llama-hook PUBLIC "${TSI_RT_LIB_DIR}")
        target_compile_features(tsi-llama-hook PRIVATE cxx_std_20)
        set_target_properties(tsi-llama-hook PROPERTIES BUILD_RPATH "${TSI_RT_LIB_DIR}")

        # Turn the hooks on inside libllama and link them in. Set from here rather than src/ because
        # the exporter targets only exist once this directory has been processed; CMake evaluates
        # target properties at generate time, so ordering is fine.
        target_compile_definitions(llama PRIVATE LLAMA_TSI_MLIR_EXPORT=1)
        target_link_libraries(llama PRIVATE tsi-llama-hook)
        set_property(TARGET llama APPEND PROPERTY BUILD_RPATH "${TSI_RT_LIB_DIR}")

        # --- decode_run: execute a compiled decode graph -------------------------------------------
        # The llama-cli hook covers prefill only (LiveGraphBuilder reconstructs a cache-free graph),
        # so this is the only thing that can run a compiled *decode* host.so. Same graph as
        # decode_cpu_check by construction - both build it with build_decode from DecodeModel.h - so
        # the CPU check and the compiled run cannot drift apart. Needs the runtime shim for
        # tsi_initialize/tsi_alloc, hence its place inside this branch.
        add_executable(decode_run tools/decode-run/decode_run.cpp)
        target_include_directories(decode_run PRIVATE
            "${TSI_MLIR_EXPORT_INCLUDE}"
            "${TSI_RT_INCLUDE_DIR}"                                  # bare "HostShimCAPI.h"
            "${MLIR_COMPILER_DIR}/third-party/runtime/include"       # its own "tsi-rt/..." includes
            "${MLIR_COMPILER_DIR}/build/_deps/runtime-build/include" # generated runtime headers
            "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-tsavorite"           # "include/TestModel.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-cpu")
        target_link_libraries(decode_run PRIVATE tsi-mlir-export llama ggml "${TSI_RT_SHIM_LIB}")
        target_link_directories(decode_run PRIVATE "${TSI_RT_LIB_DIR}")
        target_compile_features(decode_run PRIVATE cxx_std_20)
        set_target_properties(decode_run PROPERTIES BUILD_RPATH "${TSI_RT_LIB_DIR}")

        # --- test-dram-cache-persist: the KV cache design's foundational assumption ----------------
        # A DRAM memref must keep its contents between calls. Needs only the runtime shim - no ggml,
        # no llama, no exporter - because it drives a hand-written fixture, not a captured graph.
        add_executable(test-dram-cache-persist tests/dram-cache/test-dram-cache-persist.cpp)
        target_include_directories(test-dram-cache-persist PRIVATE
            "${TSI_RT_INCLUDE_DIR}"
            "${MLIR_COMPILER_DIR}/third-party/runtime/include"
            "${MLIR_COMPILER_DIR}/build/_deps/runtime-build/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src/ggml-tsavorite")   # "include/TestModel.h"
        target_link_libraries(test-dram-cache-persist PRIVATE "${TSI_RT_SHIM_LIB}")
        target_link_directories(test-dram-cache-persist PRIVATE "${TSI_RT_LIB_DIR}")
        target_compile_features(test-dram-cache-persist PRIVATE cxx_std_20)
        set_target_properties(test-dram-cache-persist PROPERTIES BUILD_RPATH "${TSI_RT_LIB_DIR}")
    else()
        message(STATUS "mlir-export: TSI runtime shim not found under ${TSI_RT_LIB_DIR}; "
                       "llama-cli whole-graph interception disabled (export library still built)")
    endif()

    # --- tsi-ggml-opt: mlir-opt-style driver, for per-op FileCheck tests -----------------------
    add_executable(tsi-ggml-opt tools/tsi-ggml-opt/tsi-ggml-opt.cpp)
    add_dependencies(tsi-ggml-opt TsiGgmlOpsIncGen)
    target_include_directories(tsi-ggml-opt PRIVATE
        "${TSI_MLIR_EXPORT_INCLUDE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dialect"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/convert"
        "${CMAKE_CURRENT_BINARY_DIR}"
        ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
    # ggml only for its headers: Exporter.h declares ggml types in its signatures.
    target_link_libraries(tsi-ggml-opt PRIVATE tsi-mlir-export ggml MLIROptLib MLIRIR MLIRParser MLIRPass
        MLIRTransforms MLIRSupport MLIRFuncDialect MLIRLinalgDialect MLIRTensorDialect
        MLIRArithDialect MLIRMathDialect)
    target_compile_features(tsi-ggml-opt PRIVATE cxx_std_17)
else()
    message(STATUS "mlir-export: MLIR not found (looked under ${MLIR_COMPILER_DIR}); "
                   "the exporter library will not be built. Configure with -DMLIR_DIR=<path to "
                   "lib/cmake/mlir> or -DMLIR_COMPILER_DIR=<mlir-compiler checkout> to enable.")
endif()

add_executable(recon_cpu_check tools/recon-cpu-check/recon_cpu_check.cpp)
target_include_directories(recon_cpu_check PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
target_link_libraries(recon_cpu_check PRIVATE ggml)
target_compile_features(recon_cpu_check PRIVATE cxx_std_20)

add_executable(ref_check tools/ref-check/ref_check.cpp)
target_link_libraries(ref_check PRIVATE llama ggml)
target_compile_features(ref_check PRIVATE cxx_std_20)

add_executable(decode_cpu_check tools/decode-cpu-check/decode_cpu_check.cpp)
target_include_directories(decode_cpu_check PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
target_link_libraries(decode_cpu_check PRIVATE ggml)
target_compile_features(decode_cpu_check PRIVATE cxx_std_20)
# --emit writes a forward.mlir, so this needs the exporter.
if (MLIR_FOUND)
    target_link_libraries(decode_cpu_check PRIVATE tsi-mlir-export)
endif()

# Stage 1 of the export test suite: builds each ggml graph, computes the CPU reference, and writes a
# self-contained case directory that the pytest runner in tests/ consumes. It lives in tests/ with
# the rest of the suite rather than under tools/, since it is test scaffolding and not a usable tool.
# Links ggml ONLY (not llama), same reason as the checks above: in a TSI build linking llama/ggml
# pulls in libggml-tsavorite and the link fails.
# Filename stem matches the target, the same way upstream tests/ derives targets with
# get_filename_component(... NAME_WE).
add_executable(test-mlir-export-cases tests/test-mlir-export-cases.cpp)
target_include_directories(test-mlir-export-cases PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
target_link_libraries(test-mlir-export-cases PRIVATE ggml)
target_compile_features(test-mlir-export-cases PRIVATE cxx_std_20)

# The generator needs the exporter library; without MLIR there is no exporter, so it is skipped.
if (MLIR_FOUND)
    target_link_libraries(test-mlir-export-cases PRIVATE tsi-mlir-export)

    # Emits a graph that reads and appends a DRAM cache, so the exporter's cache path is checked
    # independently of any model.
    add_executable(test-cache-export tests/dram-cache/test-cache-export.cpp)
    target_include_directories(test-cache-export PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
    target_link_libraries(test-cache-export PRIVATE tsi-mlir-export ggml)
    target_compile_features(test-cache-export PRIVATE cxx_std_20)

    # Weights as dense_resource constants, and bytecode as the encoding that keeps them raw.
    add_executable(test-bytecode-export tests/bytecode/test-bytecode-export.cpp)
    target_include_directories(test-bytecode-export PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
    target_link_libraries(test-bytecode-export PRIVATE tsi-mlir-export ggml)
    target_compile_features(test-bytecode-export PRIVATE cxx_std_20)
endif()

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
                         --cases-bin "$<TARGET_FILE:test-mlir-export-cases>"
                         --target ffm)
        set_tests_properties(mlir-export-suite PROPERTIES LABELS "mlir-export" TIMEOUT 1800)

        # DRAM cache persistence: two ctest entries, because the fixture has to be compiled before the
        # test can dlopen it. A ctest fixture chains them, so running the second alone still compiles.
        if (TARGET test-dram-cache-persist)
            set(DRAM_CACHE_OUT "${CMAKE_CURRENT_BINARY_DIR}/dram-cache-out")
            add_test(NAME dram-cache-compile
                     COMMAND "${TSI_VENV_PYTHON}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/compile_graph_fpga.py"
                             "${CMAKE_CURRENT_SOURCE_DIR}/tests/dram-cache/cache_persist.mlir"
                             "${DRAM_CACHE_OUT}")
            set_tests_properties(dram-cache-compile PROPERTIES
                LABELS "mlir-export" TIMEOUT 900 FIXTURES_SETUP DramCache
                ENVIRONMENT "USER_DRAM_SIZE=16348;TSI_RT_LIB_DIR=${TSI_RT_LIB_DIR}")

            add_test(NAME dram-cache-persist
                     COMMAND "$<TARGET_FILE:test-dram-cache-persist>"
                             "${DRAM_CACHE_OUT}/host/host.so")
            set_tests_properties(dram-cache-persist PROPERTIES
                LABELS "mlir-export" TIMEOUT 300 FIXTURES_REQUIRED DramCache
                ENVIRONMENT "USER_DRAM_SIZE=16348;DYLD_LIBRARY_PATH=${TSI_RT_LIB_DIR};LD_LIBRARY_PATH=${TSI_RT_LIB_DIR}")
        endif()

        # The exporter's cache path: emit a graph that reads and appends a DRAM cache, then compile it.
        # Emitting alone would not catch an invalid memref or a bad subview type, so both stages run.
        if (TARGET test-cache-export)
            set(CACHE_EXPORT_MLIR "${CMAKE_CURRENT_BINARY_DIR}/cache-export/forward.mlir")
            add_test(NAME cache-export-emit
                     COMMAND "$<TARGET_FILE:test-cache-export>" "${CACHE_EXPORT_MLIR}")
            set_tests_properties(cache-export-emit PROPERTIES
                LABELS "mlir-export" TIMEOUT 300 FIXTURES_SETUP CacheExport)

            add_test(NAME cache-export-compile
                     COMMAND "${TSI_VENV_PYTHON}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/compile_graph_fpga.py"
                             "${CACHE_EXPORT_MLIR}"
                             "${CMAKE_CURRENT_BINARY_DIR}/cache-export/out")
            set_tests_properties(cache-export-compile PROPERTIES
                LABELS "mlir-export" TIMEOUT 900 FIXTURES_REQUIRED CacheExport
                ENVIRONMENT "USER_DRAM_SIZE=16348;TSI_RT_LIB_DIR=${TSI_RT_LIB_DIR}")
        endif()

        # Weights as constants: the emit step checks the IR and the size claim, the compile step
        # checks the compiler accepts bytecode with a resource blob in it.
        if (TARGET test-bytecode-export)
            set(BYTECODE_OUT "${CMAKE_CURRENT_BINARY_DIR}/bytecode-export")
            add_test(NAME bytecode-export-emit
                     COMMAND "$<TARGET_FILE:test-bytecode-export>" "${BYTECODE_OUT}")
            set_tests_properties(bytecode-export-emit PROPERTIES
                LABELS "mlir-export" TIMEOUT 300 FIXTURES_SETUP BytecodeExport)

            add_test(NAME bytecode-export-compile
                     COMMAND "${TSI_VENV_PYTHON}"
                             "${CMAKE_CURRENT_SOURCE_DIR}/compile_graph_fpga.py"
                             "${BYTECODE_OUT}/forward.mlirbc"
                             "${BYTECODE_OUT}/out")
            set_tests_properties(bytecode-export-compile PROPERTIES
                LABELS "mlir-export" TIMEOUT 900 FIXTURES_REQUIRED BytecodeExport
                ENVIRONMENT "USER_DRAM_SIZE=16348;TSI_RT_LIB_DIR=${TSI_RT_LIB_DIR}")
        endif()
    else()
        # Never a hard error: a plain host build must not start requiring the compiler repo.
        message(STATUS "mlir-export: ${TSI_VENV_PYTHON} not found; mlir-export tests will skip "
                       "(configure with -DMLIR_COMPILER_DIR=<path> to enable)")
        add_test(NAME mlir-export-suite COMMAND ${CMAKE_COMMAND} -E echo
                 "SKIPPED: mlir-compiler venv not found at ${TSI_VENV_PYTHON}")
        set_tests_properties(mlir-export-suite PROPERTIES LABELS "mlir-export"
                             SKIP_REGULAR_EXPRESSION "SKIPPED")
    endif()

    # Per-op lowering tests: tsi-ggml-opt --convert-ggml-to-linalg | FileCheck. No Python and no TSI
    # compiler involved, so this is independent of the venv and registered separately from the two
    # tests above. llvm-lit and FileCheck come from the same LLVM build as MLIR.
    if (TARGET tsi-ggml-opt AND EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-lit"
                            AND EXISTS "${LLVM_TOOLS_BINARY_DIR}/FileCheck")
        add_test(NAME mlir-export-lit
                 COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-lit" -sv "${CMAKE_CURRENT_SOURCE_DIR}/tests/lit")
        set_tests_properties(mlir-export-lit PROPERTIES
            LABELS "mlir-export" TIMEOUT 300
            ENVIRONMENT "TSI_LIT_TOOLS_DIR=${CMAKE_BINARY_DIR}/bin;TSI_LIT_LLVM_TOOLS_DIR=${LLVM_TOOLS_BINARY_DIR};TSI_LIT_EXEC_ROOT=${CMAKE_CURRENT_BINARY_DIR}/lit-output")
    else()
        message(STATUS "mlir-export: tsi-ggml-opt, llvm-lit or FileCheck unavailable; "
                       "mlir-export-lit will skip")
        add_test(NAME mlir-export-lit COMMAND ${CMAKE_COMMAND} -E echo
                 "SKIPPED: tsi-ggml-opt, llvm-lit or FileCheck not available")
        set_tests_properties(mlir-export-lit PROPERTIES LABELS "mlir-export"
                             SKIP_REGULAR_EXPRESSION "SKIPPED")
    endif()

endif()
