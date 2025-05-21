#
# simple-ctx

set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ${TLIBS})

#
#
# simple-backend
set(TEST_TARGET simple-backend)
add_executable(${TEST_TARGET} simple-backend.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml ${TLIBS})

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
endif()
