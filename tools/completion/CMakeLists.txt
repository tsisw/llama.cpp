# llama-completion-impl: completion logic, reusable by app

set(TARGET llama-completion-impl)

add_library(${TARGET} completion.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-completion executable

set(TARGET llama-completion)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-completion-impl)
# rpath-link for transitively-needed tsi_* symbols on FPGA builds is handled
# globally via -DCMAKE_EXE_LINKER_FLAGS in tsi-pkg-build.sh (FPGA_TOOLBOX_LIB_DIR /
# FPGA_RUNTIME_LIB_DIR) -- no per-target linker flags needed here.
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
