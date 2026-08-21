# llama-cli-impl: CLI logic, reusable by app

set(TARGET llama-cli-impl)

add_library(${TARGET} cli.cpp
                      cli-client.cpp
                      cli-context.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_link_libraries(${TARGET} PUBLIC llama-server-impl llama-common ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-cli executable

set(TARGET llama-cli)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-cli-impl)
if (GGML_TSAVORITE)
if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
	set(SYSROOT_PATH_FOR_LIBRARIES $ENV{CMAKE_FIND_ROOT_PATH}/lib)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SYSROOT_PATH_FOR_LIBRARIES} -Wl,-rpath-link,${SYSROOT_PATH_FOR_LIBRARIES}")
	target_link_directories(${TARGET} PRIVATE ${SYSROOT_PATH_FOR_LIBRARIES})
endif()
endif()
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
