from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM

# Codex model prefixes that use the Responses API
CODEX_MODEL_PREFIXES = ('gpt-5', 'codex')


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif any(model_name.startswith(prefix) for prefix in CODEX_MODEL_PREFIXES):
            # Use CodexChat for gpt-5.x and codex models (Responses API)
            import swarm.llm.codex_chat  # ensure registration
            model = cls.registry.get('CodexChat', model_name)
        else: # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name)

        return model
