# HarmonyOS LLM Runtime — native entry build.
#
# Mirrors the official CANN LLM DemoNext wiring:
#   - `include/` (and `include/common`) hold the DDK headers:
#       lm_engine_model_info.h, cann_llm_engine_context.h, cann_llm_engine_executor.h,
#       llm_engine_c_api_export.h, llm_engine_context_base.h, llm_engine_executor_base.h,
#       llm_engine_return_types.h
#   - `lib64/libhiai_llm_engine.so` is the engine shared library.
# Copy both from the installed CANN Kit (see README.md).
#
# The runtime's reusable native sources live in ../../../../native and are
# compiled directly into the `entry` module.

cmake_minimum_required(VERSION 3.5.0)
project(harmony_llm_runtime)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

ADD_DEFINITIONS(-DVK_USE_PLATFORM_OHOS=1)
ADD_DEFINITIONS(-DVK_NO_PROTOTYPES=1)

# Native runtime library root (the reusable, non-NAPI layer).
set(HLLM_NATIVE_ROOT ${NATIVERENDER_ROOT_PATH}/../../../native)

include_directories(
    ${NATIVERENDER_ROOT_PATH}
    ${NATIVERENDER_ROOT_PATH}/include
    ${NATIVERENDER_ROOT_PATH}/include/common
    ${HLLM_NATIVE_ROOT}
    ${HLLM_NATIVE_ROOT}/runtime
    ${HLLM_NATIVE_ROOT}/model_package
    ${HLLM_NATIVE_ROOT}/backends/cann
)

# Imported CANN LLM Engine shared library.
add_library(hiai_llm_engine SHARED IMPORTED)
set_target_properties(hiai_llm_engine PROPERTIES IMPORTED_LOCATION
    ${NATIVERENDER_ROOT_PATH}/lib64/libhiai_llm_engine.so)

set(HLLM_NATIVE_SRC
    ${HLLM_NATIVE_ROOT}/backends/cann/CannBackend.cpp
    ${HLLM_NATIVE_ROOT}/model_package/Json.cpp
    ${HLLM_NATIVE_ROOT}/model_package/Sha256.cpp
    ${HLLM_NATIVE_ROOT}/model_package/PackageReader.cpp
)

add_library(entry SHARED LLMEngineModule.cpp ${HLLM_NATIVE_SRC})
target_link_libraries(entry PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    librawfile.z.so
    hiai_llm_engine)
