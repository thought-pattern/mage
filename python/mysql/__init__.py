"""Fail-fast marker for the optional MySQL dependency."""

raise ImportError("mysql-connector-python is required only for Mage's migration procedures")
