"""
state.py (BCQM VI)

Thin wrapper around the BCQM V ancestor state definitions.

Direct ancestry:
- `state_v_ancestor.py` is a verbatim copy of BCQM V: `bcqm_glue_axes/state.py`
- This module re-exports the ancestor API for use by bcqm_vi_spacetime.

Do not edit `state_v_ancestor.py`. If changes are needed, wrap or extend here.
"""
from .state_v_ancestor import *  # noqa: F401,F403
