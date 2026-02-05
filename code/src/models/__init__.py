"""Models package initialization."""

# Lazy imports to avoid loading heavy dependencies (torch, transformers) 
# when they're not needed

__all__ = ['ByT5Model', 'NonNeuralModel', 'NeuralBaselineModel', 'LLMModel']


def __getattr__(name):
    """Lazy load models to avoid importing heavy dependencies unnecessarily."""
    if name == 'ByT5Model':
        from .byt5 import ByT5Model
        return ByT5Model
    elif name == 'NonNeuralModel':
        from .nonneural import NonNeuralModel
        return NonNeuralModel
    elif name == 'NeuralBaselineModel':
        from .neural_baseline import NeuralBaselineModel
        return NeuralBaselineModel
    elif name == 'LLMModel':
        from .llm import LLMModel
        return LLMModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
