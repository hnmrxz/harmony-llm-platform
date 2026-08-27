/*
 * LLMEngineModule.cpp — NAPI bridge between the ArkTS UI and the native CANN
 * LLM Engine backend. Its exported surface mirrors the official sample's
 * `CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp`:
 *
 *   loadmodel(contextJson, executorJson) -> string   (load + report)
 *   modelinfer(question) -> void                     (start async generation)
 *   answerget(callback)  -> void                     (register stream callback)
 *   deinitmodel()        -> void
 *
 * Device-only: built by the HarmonyOS/DevEco CMake target with the NAPI
 * headers; links `libhiai_llm_engine.so`. See entry/src/main/cpp/CMakeLists.txt
 * for the exact include/link wiring and the DDK `include/` + `lib64/` layout.
 */

#include "napi/native_api.h"

#include <memory>
#include <string>

#include "runtime/InferenceBackend.h"

using hllm::InferenceBackend;
using hllm::GenerationCallbacks;

namespace {

std::unique_ptr<InferenceBackend> g_backend;
napi_threadsafe_function g_stream = nullptr;  // JS-side streaming token callback

// Runs on the engine worker thread for every produced token.
void CallJsToken(napi_env env, napi_value jsCallback, void* /*context*/, void* data) {
    auto* token = static_cast<std::string*>(data);
    if (jsCallback == nullptr || token == nullptr) {
        delete token;
        return;
    }
    napi_value arg;
    napi_create_string_utf8(env, token->c_str(), token->size(), &arg);
    napi_value result;
    napi_call_function(env, nullptr, jsCallback, 1, &arg, &result);
    delete token;
}

bool ReadTextFile(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
        fclose(f);
        return false;
    }
    std::string buf(static_cast<std::size_t>(len), '\0');
    if (len > 0 && fread(&buf[0], 1, static_cast<std::size_t>(len), f) !=
                       static_cast<std::size_t>(len)) {
        fclose(f);
        return false;
    }
    fclose(f);
    out = std::move(buf);
    return true;
}

napi_value LoadModel(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    auto readStr = [&](int i) {
        size_t len = 0;
        napi_get_value_string_utf8(env, args[i], nullptr, 0, &len);
        std::string s(len, '\0');
        napi_get_value_string_utf8(env, args[i], &s[0], len + 1, &len);
        return s;
    };
    if (argc < 2) {
        napi_value result;
        napi_create_string_utf8(env, "invalid arguments", NAPI_AUTO_LENGTH, &result);
        return result;
    }
    std::string contextJsonPath = readStr(0);
    std::string executorJsonPath = readStr(1);

    if (g_backend == nullptr) {
        g_backend = hllm::CreateCannEngineBackend();
    }
    if (g_backend == nullptr) {
        napi_value result;
        napi_create_string_utf8(env, "failed to create backend", NAPI_AUTO_LENGTH, &result);
        return result;
    }
    std::string contextJson, executorJson;
    bool ok = ReadTextFile(contextJsonPath, contextJson) &&
              ReadTextFile(executorJsonPath, executorJson) &&
              g_backend->LoadModel(contextJson, executorJson);

    napi_value result;
    napi_create_string_utf8(env, ok ? "模型加载完毕" : "模型加载失败", NAPI_AUTO_LENGTH, &result);
    return result;
}

napi_value AnswerGet(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1) {
        return nullptr;
    }
    if (g_stream != nullptr) {
        napi_release_threadsafe_function(g_stream, napi_tsfn_abort);
        g_stream = nullptr;
    }
    napi_value resourceName;
    napi_create_string_utf8(env, "hllm token stream", NAPI_AUTO_LENGTH, &resourceName);
    napi_create_threadsafe_function(env, args[0], nullptr, resourceName, 0, 1, nullptr,
                                    nullptr, nullptr, CallJsToken, &g_stream);
    return nullptr;
}

napi_value ModelInfer(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 1 || g_backend == nullptr) {
        return nullptr;
    }
    size_t len = 0;
    napi_get_value_string_utf8(env, args[0], nullptr, 0, &len);
    std::string prompt(len, '\0');
    napi_get_value_string_utf8(env, args[0], &prompt[0], len + 1, &len);

    GenerationCallbacks callbacks;
    callbacks.onToken = [](const std::string& token) {
        if (g_stream != nullptr) {
            napi_call_threadsafe_function(g_stream, new std::string(token), napi_tsfn_nonblocking);
        }
    };
    callbacks.onDone = [](const std::string& /*fullText*/) {};
    callbacks.onError = [](const std::string& /*error*/) {};

    g_backend->GenerateAsync(prompt, callbacks);
    return nullptr;
}

napi_value DeinitModel(napi_env env, napi_callback_info info) {
    (void)env;
    (void)info;
    if (g_backend != nullptr) {
        g_backend->Unload();
    }
    if (g_stream != nullptr) {
        napi_release_threadsafe_function(g_stream, napi_tsfn_abort);
        g_stream = nullptr;
    }
    return nullptr;
}

napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        {"loadmodel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"modelinfer", nullptr, ModelInfer, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"answerget", nullptr, AnswerGet, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"deinitmodel", nullptr, DeinitModel, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

}  // namespace

static napi_module g_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) {
    napi_module_register(&g_module);
}
