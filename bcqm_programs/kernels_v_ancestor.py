from __future__ import annotations

import numpy as np

from .state import ThreadState, BundleState, DomainAggregate


def hop_coherence_step(rng: np.random.Generator,
                       threads: ThreadState,
                       cfg_hop) -> None:
    """Single-step hop-coherence update.

    If ``threads.active`` is not None, slips are only applied to threads
    with ``active == True``; otherwise all threads are updated. This is
    where cadence gating plugs into the dynamics.
    """
    v = threads.v
    N = v.shape[0]
    q_base = getattr(cfg_hop, "q_base", None)
    if q_base is None:
        q_base = min(0.5, float(cfg_hop.k_prefactor) / 50.0)
    active = getattr(threads, "active", None)
    if active is None:
        # No cadence gating: update all threads
        slips = rng.random(N) < q_base
        v[slips] *= -1
    else:
        # Draw slips only for active threads
        slips = np.zeros(N, dtype=bool)
        if np.any(active):
            r = rng.random(N)
            slips[active] = r[active] < q_base
        v[slips] *= -1
    threads.v = v

def shared_bias_step(rng: np.random.Generator,
                     threads: ThreadState,
                     bundle: BundleState,
                     cfg_shared) -> None:
    if not cfg_shared.enabled:
        return
    v = threads.v
    N = v.shape[0]
    m = bundle.m
    if m == 0.0:
        return
    sign_m = 1.0 if m >= 0.0 else -1.0
    lambda_bias = float(cfg_shared.lambda_bias)
    aligned = v == sign_m
    flip_prob = np.clip(lambda_bias, 0.0, 1.0)
    to_flip = (~aligned) & (rng.random(N) < flip_prob)
    v[to_flip] = sign_m
    threads.v = v


def phase_lock_step(rng: np.random.Generator,
                    threads: ThreadState,
                    bundle: BundleState,
                    cfg_phase) -> None:
    if not cfg_phase.enabled:
        return
    theta = threads.theta
    N = theta.shape[0]
    theta_mean = bundle.theta_mean
    omega_0 = float(cfg_phase.omega_0)
    lambda_phase = float(cfg_phase.lambda_phase)
    noise_sigma = float(getattr(cfg_phase, "noise_sigma", 0.0))
    delta = np.angle(np.exp(1j * (theta_mean - theta)))
    theta_new = theta + omega_0 + lambda_phase * np.sin(delta)
    if noise_sigma > 0.0:
        theta_new += noise_sigma * rng.normal(size=N)
    theta_new = np.mod(theta_new, 2.0 * np.pi)
    threads.theta = theta_new
    theta_mean_new = np.angle(np.mean(np.exp(1j * theta_new)))
    delta_new = np.angle(np.exp(1j * (theta_mean_new - theta_new)))
    theta_join = float(cfg_phase.theta_join)
    theta_break = float(cfg_phase.theta_break)
    v = threads.v
    join_mask = np.abs(delta_new) < theta_join
    break_mask = np.abs(delta_new) > theta_break
    if np.any(join_mask):
        sign_m = 1.0 if bundle.m >= 0.0 else -1.0
        flip_prob = 0.1
        to_flip = join_mask & (rng.random(N) < flip_prob)
        v[to_flip] = sign_m
    if np.any(break_mask):
        rand_flip = break_mask & (rng.random(N) < 0.1)
        v[rand_flip] *= -1
    threads.v = v


def domain_aggregates(threads: ThreadState, n_domains: int) -> DomainAggregate:
    N = threads.v.shape[0]
    dom_labels = threads.domain
    sizes = np.zeros(n_domains, dtype=int)
    mags = np.zeros(n_domains, dtype=float)
    phases = np.zeros(n_domains, dtype=float)
    for d in range(n_domains):
        mask = dom_labels == d
        count = int(mask.sum())
        sizes[d] = count
        if count > 0:
            mags[d] = float(threads.v[mask].mean())
            phases[d] = float(np.angle(np.mean(np.exp(1j * threads.theta[mask]))))
        else:
            mags[d] = 0.0
            phases[d] = 0.0
    return DomainAggregate(sizes=sizes, magnetisations=mags, mean_phases=phases)


def domain_glue_step(rng: np.random.Generator,
                     threads: ThreadState,
                     cfg_domains) -> None:
    if not cfg_domains.enabled:
        return
    N = threads.v.shape[0]
    n_domains = int(getattr(cfg_domains, "n_initial_domains", 1))
    if n_domains < 1:
        n_domains = 1
    agg = domain_aggregates(threads, n_domains)
    v = threads.v.copy()
    dom = threads.domain
    lambda_dom = float(cfg_domains.lambda_domain)
    for d in range(n_domains):
        mask = dom == d
        if not np.any(mask):
            continue
        m_d = agg.magnetisations[d]
        if m_d == 0.0:
            continue
        sign_md = 1.0 if m_d >= 0.0 else -1.0
        flip_prob = np.clip(lambda_dom * abs(m_d), 0.0, 1.0)
        aligned = v[mask] == sign_md
        to_flip = (~aligned) & (np.random.random(mask.sum()) < flip_prob)
        idx = np.where(mask)[0]
        v[idx[to_flip]] = sign_md
    threads.v = v


def initialise_cadence(rng: np.random.Generator,
                       threads: ThreadState,
                       cfg_cadence) -> None:
    """Initialise per-thread cadence periods and phases.

    When cadence is disabled, we set unit periods so that every step is a tick.
    """
    N = threads.v.shape[0]
    if not getattr(cfg_cadence, "enabled", False):
        threads.T = np.ones(N, dtype=float)
        threads.phi = np.zeros(N, dtype=float)
        threads.active = np.ones(N, dtype=bool)
        return

    dist = getattr(cfg_cadence, "distribution", "lognormal")
    mean_T = float(getattr(cfg_cadence, "mean_T", 1.0))
    sigma_T = float(getattr(cfg_cadence, "sigma_T", 0.0))
    if sigma_T <= 0.0:
        T = np.full(N, max(mean_T, 1.0), dtype=float)
    else:
        if dist == "lognormal":
            # approximate lognormal parameters matching mean and sigma
            mu = np.log(mean_T**2 / np.sqrt(sigma_T**2 + mean_T**2))
            s2 = np.log(1.0 + (sigma_T**2) / (mean_T**2))
            sigma_ln = np.sqrt(max(s2, 0.0))
            T = rng.lognormal(mean=mu, sigma=sigma_ln, size=N)
        elif dist == "normal":
            T = rng.normal(loc=mean_T, scale=sigma_T, size=N)
            T = np.clip(T, 1.0, None)
        else:
            T = np.full(N, max(mean_T, 1.0), dtype=float)

    threads.T = T
    threads.phi = rng.uniform(0.0, T, size=N)
    threads.active = np.ones(N, dtype=bool)


def cadence_step(rng: np.random.Generator,
                 threads: ThreadState,
                 cfg_cadence) -> np.ndarray:
    """Advance per-thread cadence and set ``threads.active`` mask.

    - When cadence is disabled or uninitialised, all threads are active.
    - Otherwise we advance the phase ``phi`` by one unit, wrap at ``T``,
      and declare a tick where wrapping occurred.
    - A small ``lambda_cadence`` can be used to gently nudge periods
      toward the bundle mean (period synchronisation).
    """
    N = threads.v.shape[0]
    if threads.T is None or threads.phi is None or not getattr(cfg_cadence, "enabled", False):
        active = np.ones(N, dtype=bool)
        threads.active = active
        return active

    T = threads.T.astype(float)
    phi = threads.phi.astype(float)

    # advance phase by one step
    phi = phi + 1.0
    active = phi >= T
    # wrap phases that ticked
    phi[active] -= T[active]

    threads.phi = phi

    # optional period synchronisation
    lambda_cadence = float(getattr(cfg_cadence, "lambda_cadence", 0.0))
    if lambda_cadence > 0.0:
        mean_T = float(np.mean(T))
        T = T + lambda_cadence * (mean_T - T)
        T = np.clip(T, 1e-3, None)
        threads.T = T

    threads.active = active
    return active
