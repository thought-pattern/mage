"""Fail-fast marker for the optional DuckDB dependency."""

raise ImportError("duckdb is required only for Mage's migration procedures")
