"""Fail-fast marker for the optional PostgreSQL dependency."""

raise ImportError("psycopg2 is required only for Mage's migration procedures")
