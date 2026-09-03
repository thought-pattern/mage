"""Fail-fast marker for the Memgraph-only ``mgp`` host module.

Memgraph injects the real module when it loads query modules.  Local Python
execution must report that missing host capability explicitly instead of
silently using a compatibility implementation.
"""

raise ImportError("the mgp module is available only inside a Memgraph query module")
