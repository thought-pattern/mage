"""Static contract for DGL function operators."""

from typing import Any as Dynamic

def __getattr__(name: str) -> Dynamic: ...
