"""Utilities for embed worker."""

from os import environ as os_environ

from sentence_transformers import SentenceTransformer
from torch import cuda as torch_cuda


def encode_chunk(visible_gpu: int, model_id: str, texts: list[str], batch_size: int):
    """
    Runs in a spawned process. Returns (count, embeddings_as_list)
    """
    # Set device visibility BEFORE importing torch/transformers
    if visible_gpu is None:
        os_environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os_environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)

    os_environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os_environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os_environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch_cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)

    if not texts:
        return 0, []

    embs = model.encode(
        texts,
        batch_size=min(batch_size, len(texts)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    _return_value = len(texts), embs.tolist()
    return _return_value
