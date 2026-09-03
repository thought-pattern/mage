"""Static contract for Gensim's Word2Vec model."""

from typing import Any as Dynamic

class _KeyedVectors:
    index_to_key: list[int]
    vectors: list[list[float]]

class Word2Vec:
    wv: _KeyedVectors
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
