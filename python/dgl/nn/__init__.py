"""Fail-fast marker for the optional DGL dependency."""

raise ImportError("dgl is required only for Mage's link-prediction procedures")
