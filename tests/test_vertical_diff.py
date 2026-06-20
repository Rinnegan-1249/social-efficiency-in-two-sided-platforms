import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vertical_diff import (
    VerticalDiffParams, sufficient_condition_holds, profit, welfare,
    solve_qm_p, solve_qm_so,
)


def _sample_params() -> VerticalDiffParams:
    return VerticalDiffParams(qL=1.0, qH=2.0, C=1.0, B=1.0, Vbar=10.0,
                               A=1.0, beta=5.0, phi=1.0)


def test_sufficient_condition_holds_for_sample_params():
    p = _sample_params()
    assert sufficient_condition_holds(p)


def test_proposition5_excessive_variety():
    """qm_p < qm_so: proprietary platform admits MORE developers (lower
    quality threshold) than is socially optimal."""
    p = _sample_params()
    qm_p = solve_qm_p(p)
    qm_so = solve_qm_so(p)
    assert qm_p < qm_so


def test_profit_and_welfare_are_concave_on_grid():
    p = _sample_params()
    qs = np.linspace(p.qL + 1e-6, p.qH - 1e-6, 200)
    profits = np.array([profit(q, p) for q in qs])
    welfares = np.array([welfare(q, p) for q in qs])
    # second differences should be (numerically) non-positive almost everywhere for a concave fn
    d2_profit = np.diff(profits, 2)
    d2_welfare = np.diff(welfares, 2)
    assert np.mean(d2_profit <= 1e-8) > 0.95
    assert np.mean(d2_welfare <= 1e-8) > 0.95
