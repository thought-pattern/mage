"""Static contract for the optional Elasticsearch client."""

from typing import Any as Dynamic

class Elasticsearch:
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...
