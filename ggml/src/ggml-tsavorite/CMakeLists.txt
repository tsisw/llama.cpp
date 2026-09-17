message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
add_compile_options(--std=c++20)
include_directories(../ggml-cpu)
ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                        )

# TSI_REMOTE_TXE_POC build fix: ggml/CMakeLists.txt sets a directory-scoped
# CMAKE_CXX_STANDARD=17 default (non-cache `set()`, so it shadows any -D on
# the cmake command line) which CMake auto-injects as `-std=gnu++17` AFTER
# this file's own `add_compile_options(--std=c++20)` above -- GCC's "last
# -std wins" rule means the auto-injected 17 silently overrides the manual
# 20, even though this SDK's own headers (tsi-rt/utils/Profiler.h) require
# C++20 (std::source_location, std::format). Setting the target property
# directly (rather than fighting it with another raw flag) makes CMake's
# own std-flag injection emit -std=gnu++20 for this target instead.
set_target_properties(ggml-tsavorite PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

# Add compiler lib directory to link path for FFM libraries
if(DEFINED MLIR_COMPILER_DIR)
    target_link_directories(ggml-tsavorite PRIVATE "${MLIR_COMPILER_DIR}/lib")
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

