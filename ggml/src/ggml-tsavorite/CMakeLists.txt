message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
file(GLOB TLIBS  "${RUNTIME_DIR}/lib/*.so" "../../${GGML_TSI_KERNEL_DIR}/host/*.o")

ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                        )

target_link_libraries(ggml-tsavorite PRIVATE ${TLIBS} dl rt)

#if (GGML_TSAVORITE_NDEBUG)
#    add_compile_definitions(GGML_TSAVORITE_NDEBUG)
#endif()
