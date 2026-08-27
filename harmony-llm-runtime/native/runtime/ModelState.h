#pragma once

namespace hllm {

/* Model lifecycle states, matching the README state machine. */
enum class ModelState {
    Unset,
    Imported,
    Validating,
    Installed,
    Ready,
    Running,
    Error,
    Incompatible,
};

const char* ModelStateName(ModelState state);

}  // namespace hllm
