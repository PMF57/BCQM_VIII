from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ThreadState:
    v: np.ndarray
    theta: np.ndarray
    domain: np.ndarray
    history_v: Optional[np.ndarray] = None
    T: Optional[np.ndarray] = None
    phi: Optional[np.ndarray] = None
    active: Optional[np.ndarray] = None


@dataclass
class BundleState:
    X: float
    m: float
    theta_mean: float


@dataclass
class DomainAggregate:
    sizes: np.ndarray
    magnetisations: np.ndarray
    mean_phases: np.ndarray