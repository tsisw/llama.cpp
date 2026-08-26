#
# simple-ctx
set(TARGET llama-simple)
add_executable(${TARGET} simple.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${TLIBS} ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

#
if (GGML_TSAVORITE)
    #
    # tsavorite backend test cases
    if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        file(GLOB TLIBS  "${RUNTIME_DIR}/lib/*.so" "../../${GGML_TSI_KERNEL_DIR}/host/*.o")
    else()
        file(GLOB TLIBS  "${RUNTIME_DIR}/lib/*.so" "${MLIR_COMPILER_DIR}/lib/libFFMDeviceShim.so" "../../${GGML_TSI_KERNEL_DIR}/host/*.o")
    endif()
    #
    # simple-backend-tsi

    set(TEST_TARGET simple-backend-tsi)
    add_executable(${TEST_TARGET} simple-backend-tsi.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE ggml ${TLIBS}  dl rt)

    #
    # simple-chunked-repro: regression test for the chunked elementwise-op
    # (ADD/MUL/SUB/DIV) padding fix and the RMS_NORM multi-row fix (JIRA-2258
    # Gemma4 investigation)

    set(CHUNKED_REPRO_TARGET simple-chunked-repro)
    add_executable(${CHUNKED_REPRO_TARGET} simple-chunked-repro.cpp)
    target_link_libraries(${CHUNKED_REPRO_TARGET} PRIVATE ggml ${TLIBS}  dl rt)

    #
    # simple-scalar-broadcast-2d-repro: regression test for the true
    # scalar-broadcast (ne10==1) CPU-fallback fix (JIRA-2258 Gemma4
    # out_scale investigation)

    set(SCALAR2D_REPRO_TARGET simple-scalar-broadcast-2d-repro)
    add_executable(${SCALAR2D_REPRO_TARGET} simple-scalar-broadcast-2d-repro.cpp)
    target_link_libraries(${SCALAR2D_REPRO_TARGET} PRIVATE ggml ${TLIBS}  dl rt)
endif()
