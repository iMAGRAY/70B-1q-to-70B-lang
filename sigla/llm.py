from __future__ import annotations

"""Lightweight wrappers for local language models used by SIGLA."""

from .core import MissingDependencyError

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LocalLLM:
    """Simple helper for generating text with a local model.

    This class tries ``llama-cpp-python`` first for ``.ggml``/``.gguf`` weights
    and falls back to ``transformers`` if available.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048):
        self.model_path = model_path
        self.backend = None
        if Llama is not None and (model_path.endswith(".ggml") or model_path.endswith(".gguf")):
            self.backend = "llama"
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx)
        elif AutoModelForCausalLM is not None and AutoTokenizer is not None:
            self.backend = "hf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        else:
            raise MissingDependencyError("llama-cpp-python or transformers package is required")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Return the model's completion for ``prompt``."""
        if self.backend == "llama":
            result = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
            return result["choices"][0]["text"].strip()
        if self.backend == "hf":
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text[len(prompt):].strip()
        raise MissingDependencyError("no language model backend available")
