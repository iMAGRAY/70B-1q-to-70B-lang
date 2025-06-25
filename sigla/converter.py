import os
import tempfile
import tarfile
import json
from pathlib import Path
from typing import Optional

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

from .core import CapsuleStore, MissingDependencyError


def _quantize_model(model, bits: int = 8):
    """Simple dynamic quantisation of linear layers. If bits==16 –> fp16 cast."""
    if torch is None:
        raise MissingDependencyError("torch is required for quantisation")

    if bits == 8:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
    elif bits == 16:
        model = model.half()
    else:
        raise ValueError("Only 8-bit (int8) or 16-bit (fp16) quantisation supported")
    return model


def convert_to_capsulegraph(
    model_path: str,
    output_path: str,
    quant_bits: int = 8,
    device: str = "cpu",
    max_tokens: Optional[int] = None,
) -> None:
    """Convert HF модель в .capsulegraph архив.

    1. Загружаем модель и токенайзер.
    2. Квантуем модель (dynamic int8 или fp16).
    3. Извлекаем эмбеддинги токенов, создаём CapsuleStore.
    4. Сохраняем всё в tar.gz.
    """
    if AutoModel is None or AutoTokenizer is None or torch is None:
        raise MissingDependencyError("transformers[torch] packages are required")

    print(f"[convert] Loading model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    # quant
    print(f"[convert] Quantising model to {quant_bits}-bit …")
    model = _quantize_model(model, bits=quant_bits)

    # save quant model to temp file
    tmp_dir = Path(tempfile.mkdtemp())
    quant_path = tmp_dir / "quant_model.pth"
    torch.save(model.state_dict(), quant_path)

    # Build CapsuleStore from token embeddings
    print("[convert] Building token CapsuleStore …")
    with torch.no_grad():
        emb_layer = model.get_input_embeddings()  # type: ignore[attr-defined]
        weight = emb_layer.weight  # (vocab, dim)
        if device != "cpu":
            weight = weight.to("cpu")
        weight_np = weight.float().numpy()

    vocab = tokenizer.get_vocab()
    # invert to list by id order
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    vocab_size = len(inv_vocab)
    if max_tokens and vocab_size > max_tokens:
        print(f"[convert] Truncating vocab to first {max_tokens} tokens …")
        vocab_size = max_tokens
    texts = [inv_vocab[i] if i in inv_vocab else f"<unk_{i}>" for i in range(vocab_size)]

    store = CapsuleStore(model_name="token-embeddings", index_factory="Flat", auto_link_k=0)

    # Directly add vectors without re-encoding to speed up
    capsules = [{"text": t, "tags": ["token"]} for t in texts]
    vectors_slice = weight_np[:len(texts)]
    store.add_capsules(capsules, vectors=vectors_slice)  # use precomputed vectors
    # The above encodes again; for speed we could directly set index but keep simple.

    # Save store
    caps_prefix = tmp_dir / "token_caps"
    store.save(str(caps_prefix))

    # Write meta
    meta = {
        "model_path": model_path,
        "quant_bits": quant_bits,
        "vocab_size": vocab_size,
        "token_caps_prefix": "token_caps",
    }
    meta_path = tmp_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Create tar.gz capsulegraph
    print(f"[convert] Writing capsulegraph → {output_path}")
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(quant_path, arcname="quant_model.pth")
        tar.add(f"{caps_prefix}.index", arcname="token_caps.index")
        tar.add(f"{caps_prefix}.json", arcname="token_caps.json")
        tar.add(meta_path, arcname="meta.json")

    print("[convert] Done.") 