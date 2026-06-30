"""Compatibility shim.

The proprietary prompt-building logic has been removed for the public release.
``SCHEMA`` (loaded from ``param_schema.json``) and ``get_param_default`` are
re-exported from the dummy ``patient_simulator`` module so existing imports
(e.g. ``from ...src.param_prompts import SCHEMA``) keep working.
See ``README.md`` in this directory.
"""
from src.patient_simulator import (
    SCHEMA,
    get_param_default,
)

__all__ = ["SCHEMA", "get_param_default"]
