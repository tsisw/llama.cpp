# Host-side CPU checks for the whole-graph reconstruction (no FPGA needed). Build with a non-TSI
# host cmake config, not the cross-compiled fpga package build (there, linking ggml pulls in
# libggml-tsavorite and the link fails):
#
#   cmake -B build-host -DLLAMA_CURL=OFF -DGGML_TSAVORITE_TARGET=posix
#   cmake --build build-host --target ref_check recon_cpu_check
#
# ref_check prints llama.cpp's prompt ids and greedy next token; recon_cpu_check rebuilds build_layer
# on CPU for the same ids. Feed ref_check's ids to recon_cpu_check and compare the argmax.
#   ./build-host/bin/ref_check        model.gguf "hello world"     -> ids + reference token
#   ./build-host/bin/recon_cpu_check  model.gguf <id0> <id1> ...   -> recon argmax per column

add_executable(recon_cpu_check recon_cpu_check.cpp)
target_link_libraries(recon_cpu_check PRIVATE ggml)
target_compile_features(recon_cpu_check PRIVATE cxx_std_20)

add_executable(ref_check ref_check.cpp)
target_link_libraries(ref_check PRIVATE llama ggml)
target_compile_features(ref_check PRIVATE cxx_std_20)
