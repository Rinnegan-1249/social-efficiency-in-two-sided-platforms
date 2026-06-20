import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.competition import (
    CompetitionParams, n_competing, n_monopoly, welfare_gain, t_H, t_L, find_valid_interval,
)


def test_monopoly_induces_more_developer_entry_than_competition():
    p = CompetitionParams(A=2.0, beta=0.5, phi=0.3)
    assert n_monopoly(p) > n_competing(p)


def test_welfare_gain_zero_at_t_H():
    p = CompetitionParams(A=2.0, beta=0.5, phi=0.3)
    assert math.isclose(welfare_gain(t_H(p), p), 0.0, abs_tol=1e-8)


def test_interval_non_empty_for_various_beta():
    for beta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        p = CompetitionParams(A=1.5, beta=beta, phi=0.4)
        lo, hi = find_valid_interval(p)
        assert lo <= hi, f"empty interval at beta={beta}"


def test_welfare_gain_decreasing_in_t():
    p = CompetitionParams(A=2.0, beta=0.5, phi=0.3)
    assert welfare_gain(0.0, p) > welfare_gain(10.0, p)


def test_monopoly_preferred_within_interval():
    p = CompetitionParams(A=2.0, beta=0.5, phi=0.3)
    lo, hi = find_valid_interval(p)
    mid = (lo + hi) / 2
    assert welfare_gain(mid, p) >= 0
