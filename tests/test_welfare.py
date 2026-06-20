import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import example1
from src.equilibria import solve_equilibria, make_F, make_H_inv
from src.welfare import welfare


def test_welfare_matches_numeric_integration():
    B, C, epsilon_F, epsilon_H = 1.2, 0.8, 1.0, 2.0
    m = example1(alpha=0.2, sigma=0.4, c=1.0)
    n, theta_m = 5.0, 3.0

    F = make_F(B, epsilon_F)
    H_inv = make_H_inv(C, epsilon_H)
    f = lambda theta: B * epsilon_F * theta ** (epsilon_F - 1)
    h = lambda phi: (1.0 / C) ** epsilon_H * epsilon_H * phi ** (epsilon_H - 1)

    user_term_numeric, _ = quad(lambda th: th * f(th), 0, theta_m)
    dev_term_numeric, _ = quad(lambda phi: phi * h(phi), 0, H_inv(n))

    expected = m.V(n) * F(theta_m) - user_term_numeric - dev_term_numeric
    actual = welfare(m, n, theta_m, epsilon_F, epsilon_H, B, C)
    assert math.isclose(actual, expected, rel_tol=1e-4)


def test_proposition4_ii_welfare_ratio_near_known_bound_as_sigma_to_1():
    """As sigma -> 1 (alpha fixed), W(n_p)/W(n_fe) -> 3/4 (paper, Prop 4.ii)."""
    alpha = 0.05
    epsilon_F, epsilon_H = 1.0, float("inf")
    sigmas = [0.9, 0.95, 0.99, 0.999]
    ratios = []
    for sigma in sigmas:
        m = example1(alpha=alpha, sigma=sigma, c=1.0)
        res = solve_equilibria(m, epsilon_F=epsilon_F, epsilon_H=epsilon_H, require_valid=False)
        ratios.append(welfare(m, res.n_p, res.theta_m_p, epsilon_F, epsilon_H) /
                       welfare(m, res.n_fe, res.theta_m_fe, epsilon_F, epsilon_H))
    # ratio should be moving toward 0.75 as sigma -> 1
    assert abs(ratios[-1] - 0.75) < abs(ratios[0] - 0.75)
