#
# mnist-common

set(TEST_TARGET mnist-common)
add_library(${TEST_TARGET} STATIC mnist-common.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common ${TLIBS})

#
# mnist-eval

set(TEST_TARGET mnist-eval)
add_executable(${TEST_TARGET} mnist-eval.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common ${TLIBS})

#
# mnist-train

set(TEST_TARGET mnist-train)
add_executable(${TEST_TARGET} mnist-train.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common mnist-common ${TLIBS})
