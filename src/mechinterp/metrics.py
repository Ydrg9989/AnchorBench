"""Distributional anchoring metrics over integer distributions P(y) for y in 0..100."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _safe(p: np.ndarray) -> np.ndarray:
    """Ensure no zeros for log operations."""
    return np.clip(p, EPS, None)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with epsilon smoothing."""
    p, q = _safe(p / p.sum()), _safe(q / q.sum())
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL divergence: 0.5 * (KL(p||q) + KL(q||p))."""
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""
    p_n, q_n = p / p.sum(), q / q.sum()
    m = 0.5 * (p_n + q_n)
    return 0.5 * kl_divergence(p_n, m) + 0.5 * kl_divergence(q_n, m)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance."""
    return 0.5 * float(np.abs(p / p.sum() - q / q.sum()).sum())


def wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """Wasserstein-1 (earth mover's) distance on integer support 0..100."""
    cdf_p = np.cumsum(p / p.sum())
    cdf_q = np.cumsum(q / q.sum())
    return float(np.abs(cdf_p - cdf_q).sum())


def expected_value(p: np.ndarray) -> float:
    """E[Y] where Y in {0, ..., len(p)-1}."""
    p_n = p / p.sum()
    return float(np.sum(np.arange(len(p_n)) * p_n))


def delta_ev(p_anch: np.ndarray, p_ctrl: np.ndarray) -> float:
    """Shift in expected value: E[Y|anchor] - E[Y|control]."""
    return expected_value(p_anch) - expected_value(p_ctrl)


def nai(p_low: np.ndarray, p_high: np.ndarray, anchor_gap: float = 60.0) -> float:
    """Normalized Anchoring Index = (EV_high - EV_low) / anchor_gap."""
    return (expected_value(p_high) - expected_value(p_low)) / anchor_gap


def compute_all(
    p_ctrl: np.ndarray, p_low: np.ndarray, p_high: np.ndarray,
    anchor_gap: float = 60.0,
) -> dict[str, float]:
    """Compute all metrics for a (control, low, high) triplet."""
    return {
        "ev_ctrl": expected_value(p_ctrl),
        "ev_low": expected_value(p_low),
        "ev_high": expected_value(p_high),
        "delta_ev_low": delta_ev(p_low, p_ctrl),
        "delta_ev_high": delta_ev(p_high, p_ctrl),
        "kl_low": kl_divergence(p_low, p_ctrl),
        "kl_high": kl_divergence(p_high, p_ctrl),
        "sym_kl_low": symmetric_kl(p_low, p_ctrl),
        "sym_kl_high": symmetric_kl(p_high, p_ctrl),
        "js_low": js_divergence(p_low, p_ctrl),
        "js_high": js_divergence(p_high, p_ctrl),
        "tvd_low": tvd(p_low, p_ctrl),
        "tvd_high": tvd(p_high, p_ctrl),
        "w1_low": wasserstein1(p_low, p_ctrl),
        "w1_high": wasserstein1(p_high, p_ctrl),
        "nai": nai(p_low, p_high, anchor_gap),
    }
