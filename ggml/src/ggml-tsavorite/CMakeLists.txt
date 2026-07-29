message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
add_compile_options(--std=c++20)
include_directories(../ggml-cpu)
# whole-graph interception (tsi_wholegraph.cpp) reuses the exporter emitter
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/mlir-export/include)
ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                         tsi_wholegraph.cpp
                        )

# Add compiler lib directory to link path for FFM libraries
if(DEFINED MLIR_COMPILER_DIR)
    target_link_directories(ggml-tsavorite PRIVATE "${MLIR_COMPILER_DIR}/lib")
endif()

# The whole-graph exporter is now an MLIR C++ library, so tsi_wholegraph.cpp needs it at link time.
# There is no aarch64 build of MLIR, so the fpga target CANNOT link this - that is expected and
# accepted. Warn with the reason rather than leaving a bare undefined-symbol error to interpret.
if (TARGET tsi-mlir-export)
    target_link_libraries(ggml-tsavorite PRIVATE tsi-mlir-export)
    target_link_libraries(decode_run PRIVATE tsi-mlir-export)
else()
    message(WARNING
        "ggml-tsavorite: tsi-mlir-export target not present, so TSI_WHOLEGRAPH capture will not "
        "link. The exporter is an MLIR C++ library now and MLIR is host-only here; the fpga/aarch64 "
        "target cannot build it. Use a posix build for whole-graph export.")
endif()


if (NOT ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
target_link_libraries(ggml-tsavorite PRIVATE ${TLIBS} dl rt)
else()

message(STATUS "OLLAMA = ${OLLAMA}")

if(OLLAMA)
target_link_libraries(ggml-tsavorite
  PRIVATE
    -Wl,--no-as-needed
    ${CMAKE_BINARY_DIR}/lib/ollama/libggml-cpu.so
    ${TLIBS}
    dl
    rt
)
else()
target_link_libraries(ggml-tsavorite
  PRIVATE
    -Wl,--no-as-needed
    ggml-cpu
    ${TLIBS}
    dl
    rt
)
endif()
set_target_properties(ggml-tsavorite PROPERTIES
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib/ollama"
)
endif()

# Standalone runner for the compiled KV-cache decode graph. Needs the TSI runtime (tsi_alloc, via
# TLIBS) and, at run time, the compiled decode host.so. EXCLUDE_FROM_ALL so it never blocks the
# package build - build on demand: cmake --build <dir> --target decode_run
add_executable(decode_run EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/mlir-export/tools/decode-run/decode_run.cpp)
target_include_directories(decode_run PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../examples/mlir-export/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu)
target_link_libraries(decode_run PRIVATE llama ggml-tsavorite ggml ggml-base ggml-cpu ${TLIBS} dl rt)
target_compile_features(decode_run PRIVATE cxx_std_20)

