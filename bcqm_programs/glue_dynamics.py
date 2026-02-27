"""
glue_dynamics.py (BCQM VI)

Wrapper around BCQM V glue-axis kernels.

Direct ancestry:
- `kernels_v_ancestor.py` is a verbatim copy of BCQM V: `bcqm_glue_axes/kernels.py`

Rule:
- Call the ported kernel functions using the SAME internal parameter names as in BCQM V.
- Any renaming/translation occurs only in the compat/import layer (compat_v5_v6.py), not here.
"""
from __future__ import annotations

# Re-export the V kernel functions and any helper types/constants.
# We intentionally keep this permissive during the port stage.
from .kernels_v_ancestor import *  # noqa: F401,F403
