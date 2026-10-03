"""
nodes/griptape_compat.py
Forwards to ibkvit_core.griptape_compat for backwards compatibility.
"""

from ibkvit_core.griptape_compat import (
    DataNode,
    IS_GRIPTAPE_ENVIRONMENT,
    Parameter,
    ParameterMode,
)

__all__ = [
    "ParameterMode",
    "Parameter",
    "DataNode",
    "IS_GRIPTAPE_ENVIRONMENT",
]
