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
# When `lib64/libhiai_llm_engine.so` is absent, a no-op engine stub is compiled
# instead so the SDK/toolchain path can still be validated (host/CI build).

cmake_minimum_required(VERSION 3.5.0)
project(harmony_llm_runtime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

ADD_DEFINITIONS(-DVK_USE_PLATFORM_OHOS=1)
ADD_DEFINITIONS(-DVK_NO_PROTOTYPES=1)

# Native runtime library root. This CMake file is at
#   <repo>/harmony-llm-runtime/entry/src/main/cpp/CMakeLists.txt
# so the reusable native layer is four directories up at <repo>/harmony-llm-runtime/native.
set(HLLM_NATIVE_ROOT ${NATIVERENDER_ROOT_PATH}/../../../../native)

include_directories(
    ${NATIVERENDER_ROOT_PATH}
    ${NATIVERENDER_ROOT_PATH}/include
    ${NATIVERENDER_ROOT_PATH}/include/common
    ${HLLM_NATIVE_ROOT}
    ${HLLM_NATIVE_ROOT}/runtime
    ${HLLM_NATIVE_ROOT}/model_package
    ${HLLM_NATIVE_ROOT}/backends/cann
)

set(HLLM_ENGINE_LIB ${NATIVERENDER_ROOT_PATH}/lib64/libhiai_llm_engine.so)

set(HLLM_NATIVE_SRC
    ${HLLM_NATIVE_ROOT}/backends/cann/CannBackend.cpp
    ${HLLM_NATIVE_ROOT}/model_package/Json.cpp
    ${HLLM_NATIVE_ROOT}/model_package/Sha256.cpp
    ${HLLM_NATIVE_ROOT}/model_package/PackageReader.cpp
    ${HLLM_NATIVE_ROOT}/model_package/ZipReader.cpp
    ${HLLM_NATIVE_ROOT}/model_package/Inflate.cpp
    ${HLLM_NATIVE_ROOT}/runtime/ModelManager.cpp
)

set(HLLM_EXTRA_LIBS libace_napi.z.so libhilog_ndk.z.so librawfile.z.so)

if(EXISTS ${HLLM_ENGINE_LIB})
    # Real CANN Kit present: link the engine.
    add_library(hiai_llm_engine SHARED IMPORTED)
    set_target_properties(hiai_llm_engine PROPERTIES IMPORTED_LOCATION ${HLLM_ENGINE_LIB})
    list(APPEND HLLM_EXTRA_LIBS hiai_llm_engine)
else()
    # No CANN Kit: compile the no-op engine stub so the module still links.
    list(APPEND HLLM_NATIVE_SRC ${HLLM_NATIVE_ROOT}/runtime/LLMEngineStub.cpp)
endif()

add_library(entry SHARED LLMEngineModule.cpp ${HLLM_NATIVE_SRC})
target_link_libraries(entry PUBLIC ${HLLM_EXTRA_LIBS})
