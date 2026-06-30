"""Compatibility shim.

The proprietary graph-node implementation has been removed for the public
release. The names below are re-exported from the dummy ``patient_simulator``
module so existing imports (e.g. ``from ...src.nodes import STOP_TOKEN``) keep
working. See ``README.md`` in this directory.
"""
from src.patient_simulator import (
    DEFAULT_MODEL_MAP,
    STOP_TOKEN,
    VERBOSITY_CATEGORY_MAP,
    ingest_node,
)

__all__ = ["DEFAULT_MODEL_MAP", "STOP_TOKEN", "VERBOSITY_CATEGORY_MAP", "ingest_node"]
