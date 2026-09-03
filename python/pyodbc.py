"""Fail-fast marker for the optional ODBC dependency."""

raise ImportError("pyodbc is required only for Mage's migration procedures")
