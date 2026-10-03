"""Utilities for embed worker."""

from os import environ as os_environ

from sentence_transformers import SentenceTransformer
from torch import cuda as torch_cuda

# The encoder this spawned worker process loads once for the request that owns it. The process, and with it the model's
# device memory, ends when that request shuts its pool down.
MODEL = "model"
worker_model: dict[str, SentenceTransformer] = {}


def load_model(visible_gpu: int, model_id: str) -> None:
    """
    Pool initializer: runs once in each spawned worker process and loads the encoder every later chunk reuses.
    """
    # Restrict device visibility before CUDA is initialized in this process.
    os_environ["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)

    os_environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os_environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os_environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch_cuda.is_available() else "cpu"
    worker_model[MODEL] = SentenceTransformer(model_id, device=device)


def encode_chunk(texts: list[str], batch_size: int):
    """
    Runs in a worker process prepared by load_model. Returns (count, embeddings_as_list)
    """
    model = worker_model.get(MODEL, False)
    if model is False:
        raise RuntimeError("Embedding worker has no loaded model; load_model must run as the pool initializer.")

    if not texts:
        empty_result = 0, []
        return empty_result

    embs = model.encode(
        texts,
        batch_size=min(batch_size, len(texts)),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    computed_return_value = len(texts), embs.tolist()
    return computed_return_value
