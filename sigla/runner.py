from __future__ import annotations

"""Utility to load and run a `.capsulegraph` archive.

Example
-------
$ python -m sigla.runner mymodel.capsulegraph --prompt "Hello" --device cuda
"""

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Tuple, Optional

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
) -> Tuple["torch.nn.Module", "transformers.PreTrainedTokenizer", CapsuleStore]:
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
# Simple CLI
# ---------------------------------------------------------------------------

def _cli():  # noqa: D401
    parser = argparse.ArgumentParser(description="Run .capsulegraph archive")
    parser.add_argument("archive", help="Path to .capsulegraph")
    parser.add_argument("--prompt", help="Prompt to generate", default="Hello, world!")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    model, tok, _ = load_capsulegraph(args.archive, device=args.device)

    print("[runner] Generating…")
    import torch

    input_ids = tok(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        output = model.generate(
            **input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.9,
        )
    text = tok.decode(output[0], skip_special_tokens=True)
    print("=== Generated ===\n", text)


if __name__ == "__main__":
    _cli() 