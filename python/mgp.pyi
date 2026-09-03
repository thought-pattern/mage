"""Type contract for Memgraph's host-provided ``mgp`` module.

The real module is injected by Memgraph when procedures are loaded.  This
stub describes the small Python-facing surface used by Mage so local static
analysis can validate procedure code without pretending that the host module
is available to ordinary Python processes.
"""

from typing import Any as Dynamic
from typing import TypeVar

type Any = Dynamic
type Nullable[T] = Dynamic
type Map[K, V] = Dynamic
type List[T] = Dynamic
type Number = Dynamic
type ProcCtx = Dynamic
type Vertices = Dynamic

T = TypeVar("T")

class Label:
    name: str
    def __getattr__(self, name: str) -> Dynamic: ...

class EdgeType:
    name: str
    def __init__(self, name: str) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...

class Vertex:
    def __getattr__(self, name: str) -> Dynamic: ...

class Edge:
    def __getattr__(self, name: str) -> Dynamic: ...

class Path:
    def __init__(self, *vertices: Dynamic) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...

class Logger:
    def __init__(self, *args: Dynamic, **kwargs: Dynamic) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...

class Record:
    def __init__(self, *args: Dynamic, **values: Dynamic) -> None: ...
    def __getattr__(self, name: str) -> Dynamic: ...

def function(procedure: T) -> T: ...
def read_proc(procedure: T) -> T: ...
def write_proc(procedure: T) -> T: ...
def add_batch_read_proc(*procedures: T) -> T: ...
