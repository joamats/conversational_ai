"""Compatibility shim.

``SimulatorConfig`` now lives in the dummy ``patient_simulator`` module. It is
re-exported here so existing imports (e.g. ``from ...src.config_wrapper import
SimulatorConfig``) keep working. See ``README.md`` in this directory.
"""
from src.patient_simulator import (
    SimulatorConfig,
    _DEFAULTS,
)

__all__ = ["SimulatorConfig", "_DEFAULTS"]
