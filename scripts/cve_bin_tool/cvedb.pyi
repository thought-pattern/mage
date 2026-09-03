"""Static contract for the optional CVE database dependency."""

from typing import Any as Dynamic

class CVEDB:
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...
