from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .glue import GlueParams, sync_strength


def choose_targets(
    rng: np.random.Generator,
    candidates: List[List[int]],
    allow_new: bool,
    g: GlueParams,
    preferred_existing: int | None,
) -> List[Tuple[str, int | None]]:
    """Placeholder policy for a controllable lockstep/cross-link signal.

    Returns per-thread choice:
      - ("existing", node_id) or ("new", None)

    Design intent (scaffold only):
      - Increasing glue synchrony increases re-use of existing events (cross-links).
      - A *capped* tailgating probability encourages co-selection without forcing
        total collapse to a single hub.
    """
    p_sync = sync_strength(g)

    # Existing-vs-new:
    # Keep some new-event creation even at high synchrony to prevent stalling.
    p_existing = 0.2 + 0.6 * p_sync  # max 0.8
    if not allow_new:
        p_existing = 1.0

    # Tailgating cap: even at p_sync=1, don't force 100% convergence.
    p_tailgate = min(0.95, p_sync)

    out: List[Tuple[str, int | None]] = []
    for i in range(len(candidates)):
        use_existing = (rng.random() < p_existing) and (len(candidates[i]) > 0)

        if use_existing:
            if (
                preferred_existing is not None
                and preferred_existing in candidates[i]
                and (rng.random() < p_tailgate)
            ):
                out.append(("existing", preferred_existing))
            else:
                out.append(("existing", int(rng.choice(candidates[i]))))
        else:
            out.append(("new", None))

    return out
