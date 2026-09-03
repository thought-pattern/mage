"""Static contract for DGL data loaders."""

from typing import Any as Dynamic

def __getattr__(name: str) -> Dynamic: ...
