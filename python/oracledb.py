"""Fail-fast marker for the optional Oracle dependency."""

raise ImportError("oracledb is required only for Mage's migration procedures")
