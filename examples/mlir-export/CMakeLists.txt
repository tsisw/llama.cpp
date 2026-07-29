# Whole-graph ggml -> linalg MLIR exporter, its host-side CPU checks, and the export test suite.
#
# Build with a non-TSI host cmake config, not the cross-compiled fpga package build (there, linking
# ggml pulls in libggml-tsavorite and the link fails):
#
#   cmake -B build-host -DLLAMA_CURL=OFF -DGGML_TSAVORITE_TARGET=posix
#   cmake --build build-host --target ref_check recon_cpu_check decode_cpu_check mlir-export-cases
#
# ref_check prints llama.cpp's prompt ids and greedy next token; recon_cpu_check rebuilds build_layer
# on CPU for the same ids; decode_cpu_check checks the KV-cache decode step reproduces the prefill.
#   ./build-host/bin/ref_check          model.gguf "hello world"     -> ids + reference token
#   ./build-host/bin/recon_cpu_check    model.gguf <id0> <id1> ...   -> recon argmax per column
#   ./build-host/bin/decode_cpu_check   model.gguf <id0> <id1> ...   -> decode-vs-prefill MATCH per step
#
# Layout: include/tsi/{export,graph} public headers, src/export the exporter library, tools/<name>
# one executable each. The .py and .sh drivers stay at this directory's top level because
# tsi-pkg-build.sh copies them from here by name and wholegraph.sh/decode.sh require
# compile_graph_fpga.py to sit next to them.

set(TSI_MLIR_EXPORT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

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

# Case generator for the export test suite. Links ggml ONLY (not llama), same reason as the
# checks above: in a TSI build linking llama/ggml pulls in libggml-tsavorite and the link fails.
add_executable(mlir-export-cases tools/mlir-export-cases/mlir_export_cases.cpp)
target_include_directories(mlir-export-cases PRIVATE "${TSI_MLIR_EXPORT_INCLUDE}")
target_link_libraries(mlir-export-cases PRIVATE ggml)
target_compile_features(mlir-export-cases PRIVATE cxx_std_20)

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

        # Structural check against committed golden IR. Compiles nothing, so it is seconds rather
        # than minutes, and it is what protects the emitters whose numerics are insensitive to a
        # structural change. Separate test so a numeric failure and an IR change are distinguishable.
        add_test(NAME mlir-export-golden
                 COMMAND "${TSI_VENV_PYTHON}" -m pytest -v
                         "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ir_golden.py"
                         --cases-bin "$<TARGET_FILE:mlir-export-cases>")
        set_tests_properties(mlir-export-golden PROPERTIES LABELS "mlir-export" TIMEOUT 600)
    else()
        # Never a hard error: a plain host build must not start requiring the compiler repo.
        message(STATUS "mlir-export: ${TSI_VENV_PYTHON} not found; mlir-export tests will skip "
                       "(configure with -DMLIR_COMPILER_DIR=<path> to enable)")
        add_test(NAME mlir-export-suite COMMAND ${CMAKE_COMMAND} -E echo
                 "SKIPPED: mlir-compiler venv not found at ${TSI_VENV_PYTHON}")
        set_tests_properties(mlir-export-suite PROPERTIES LABELS "mlir-export"
                             SKIP_REGULAR_EXPRESSION "SKIPPED")
        add_test(NAME mlir-export-golden COMMAND ${CMAKE_COMMAND} -E echo
                 "SKIPPED: mlir-compiler venv not found at ${TSI_VENV_PYTHON}")
        set_tests_properties(mlir-export-golden PROPERTIES LABELS "mlir-export"
                             SKIP_REGULAR_EXPRESSION "SKIPPED")
    endif()
endif()
