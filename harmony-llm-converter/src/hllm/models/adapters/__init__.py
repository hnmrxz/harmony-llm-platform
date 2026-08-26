from hllm.models.adapters.qwen import QwenAdapter
from hllm.models.registry import ModelRegistry


def default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(QwenAdapter())
    return registry


__all__ = ["QwenAdapter", "default_registry"]
