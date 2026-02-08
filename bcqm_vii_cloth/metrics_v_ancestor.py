from __future__ import annotations

from typing import Dict

import numpy as np


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n == 0:
        return np.array([1.0])
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(fx * np.conjugate(fx), nfft)[:n]
    acf = acf / acf[0]
    return acf


def compute_lockstep_metrics(m_all: np.ndarray,
                             dX_all: np.ndarray) -> Dict[str, float]:
    m_all = np.asarray(m_all, dtype=float)
    dX_all = np.asarray(dX_all, dtype=float)
    L_inst = float(np.mean(np.abs(m_all)))
    acfs = []
    for e in range(m_all.shape[0]):
        acf = _autocorrelation(m_all[e, :])
        acfs.append(acf)
    acfs = np.vstack(acfs)
    acf_mean = acfs.mean(axis=0)
    ell_lock = float(np.sum(np.maximum(acf_mean, 0.0)))
    mean_dX = float(np.mean(dX_all))
    std_dX = float(np.std(dX_all))
    Q_clock = float(abs(mean_dX) / std_dX) if std_dX > 0 else 0.0
    return {
        "L_inst": L_inst,
        "ell_lock": ell_lock,
        "Q_clock": Q_clock,
    }
