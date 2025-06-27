from __future__ import annotations

"""Utility to load and run a `.capsulegraph` archive.

Example
-------
$ python -m sigla.runner mymodel.capsulegraph --prompt "Hello" --device cuda
"""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Any

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
except Exception:
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore

from .core import CapsuleStore, MissingDependencyError


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_capsulegraph(
    archive_path: str | Path,
    device: str = "cpu",
) -> Tuple[Any, Any, CapsuleStore]:
    """Return quantised model, tokenizer and token CapsuleStore.

    Parameters
    ----------
    archive_path : str | Path
        Path to `.capsulegraph` tar.gz archive.
    device : str
        cpu / cuda / mps.
    """
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise MissingDependencyError("transformers[torch] packages are required to run capsulegraph")

    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    tmp_dir = Path(tempfile.mkdtemp())
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(tmp_dir)

    # Required files
    meta_file = tmp_dir / "meta.json"
    quant_file = tmp_dir / "quant_model.pth"
    token_caps_prefix = tmp_dir / "token_caps"  # .index/.json

    if not meta_file.exists() or not quant_file.exists():
        raise RuntimeError("Invalid capsulegraph: missing meta or quant model")

    meta = json.loads(meta_file.read_text())
    model_path = meta.get("model_path")
    quant_bits = int(meta.get("quant_bits", 8))

    if not model_path:
        raise RuntimeError("meta.json must contain 'model_path'")

    print(f"[runner] Loading tokenizer from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[runner] Loading base model from {model_path} …")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True) if AutoModelForCausalLM else AutoModel.from_pretrained(model_path, trust_remote_code=True)  # type: ignore

    # Quantise as recorded
    if quant_bits == 8:
        base_model = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
            base_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
    elif quant_bits == 16:
        base_model = base_model.half()

    print("[runner] Loading quantised weights …")
    state_dict = torch.load(quant_file, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[runner] Warning: missing keys: {missing[:5]} …")
    if unexpected:
        print(f"[runner] Warning: unexpected keys: {unexpected[:5]} …")

    base_model.to(device)
    base_model.eval()

    print("[runner] Loading token CapsuleStore …")
    store = CapsuleStore()
    store.load(str(token_caps_prefix))

    return base_model, tokenizer, store


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def generate_with_prefix(
    model: Any,
    tokenizer,
    prompt: str,
    prefix_tokens: Optional[List[int]] = None,
    *,
    max_tokens: int = 128,
    temperature: float = 0.9,
    device: str = "cpu",
) -> str:
    """Generate text optionally seeding KV-cache with *prefix_tokens*.

    Implementation strategy:  simply concatenate the prefix token IDs with
    the prompt token IDs – while this is not *true* KV-cache injection, the
    effect on autoregressive models is equivalent and universally supported
    by HF generate().
    """

    import torch

    # Encode prompt to ids (without special tokens)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if prefix_tokens:
        input_ids = torch.tensor([prefix_tokens + prompt_ids], device=device)
    else:
        input_ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Simple CLI
# ---------------------------------------------------------------------------

def _cli():  # noqa: D401
    parser = argparse.ArgumentParser(description="Run .capsulegraph archive")
    parser.add_argument("archive", help="Path to .capsulegraph")
    parser.add_argument("--prompt", help="Prompt to generate", default="Hello, world!")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefix", help="JSON list of token IDs or comma-separated ints", default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    model, tok, _ = load_capsulegraph(args.archive, device=args.device)

    # Parse prefix tokens if any
    prefix_tokens: Optional[List[int]] = None
    if args.prefix:
        import json
        try:
            if args.prefix.strip().startswith("["):
                prefix_tokens = json.loads(args.prefix)
            else:
                prefix_tokens = [int(x) for x in args.prefix.split(",") if x.strip()]
        except Exception as e:
            print(f"[runner] Failed to parse prefix tokens: {e}")
            return

    print("[runner] Generating…")
    text = generate_with_prefix(
        model,
        tok,
        args.prompt,
        prefix_tokens=prefix_tokens,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    print("=== Generated ===\n", text)


if __name__ == "__main__":
    _cli()
