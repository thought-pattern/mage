"""Static contract for optional DGL runtime support."""

from typing import Any as Dynamic

AddReverse: Dynamic

class graph:
    canonical_etypes: list[tuple[str, str, str]]
    ntypes: list[str]
    device: Dynamic
    idtype: Dynamic
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
    def __call__(self, *args: Dynamic, **kwargs: Dynamic) -> Dynamic: ...
    def __getitem__(self, key: Dynamic) -> Dynamic: ...
    def __getattr__(self, name: str) -> Dynamic: ...

class heterograph(graph): ...

function: Dynamic
dataloading: Dynamic
nn: Dynamic
