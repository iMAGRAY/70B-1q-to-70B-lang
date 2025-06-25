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


# ---------------------------------------------------------------------------
# GitHub Models integration (Codestral / др.).  Requires ``mistralai`` SDK.
# ---------------------------------------------------------------------------


class GitHubLLM:  # noqa: D401
    """Wrapper around GitHub Models inference endpoint (public preview).

    Пример использования::

        from sigla.llm import GitHubLLM
        llm = GitHubLLM()
        answer = llm.chat("Назови столицу Франции")
    """

    _ENDPOINT = "https://models.github.ai/inference"

    def __init__(self, model: str = "mistral-ai/Codestral-2501", temperature: float = 0.7):
        try:
            from mistralai import Mistral, UserMessage, SystemMessage  # type: ignore
        except Exception as e:  # pragma: no cover – optional dependency
            raise MissingDependencyError("mistralai package is required: pip install mistralai") from e

        import os

        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise MissingDependencyError("GITHUB_TOKEN environment variable not set")

        self._model_name = model
        self._temperature = temperature
        self._client = Mistral(api_key=token, server_url=self._ENDPOINT)

        # cache message classes for fast access
        self._UserMessage = UserMessage  # type: ignore
        self._SystemMessage = SystemMessage  # type: ignore

    # ------------------------------------------------------------------
    # Basic chat
    # ------------------------------------------------------------------

    def chat(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str:
        """Synchronous chat-completion call.  Returns first choice text."""

        messages = []
        if system:
            messages.append(self._SystemMessage(content=system))
        messages.append(self._UserMessage(content=prompt))

        resp = self._client.chat.complete(
            model=self._model_name,
            messages=messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
        return resp.choices[0].message.content

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, prompt: str, system: str | None = None):
        """Generator yielding chunks of text as they arrive."""

        try:
            from mistralai import UserMessage, SystemMessage  # type: ignore
        except Exception:
            raise MissingDependencyError("mistralai package is required")

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(UserMessage(content=prompt))

        for chunk in self._client.chat.stream(model=self._model_name, messages=messages):
            if chunk.data.choices:
                delta = chunk.data.choices[0].delta.content or ""
                if delta:
                    yield delta
