import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import example1
from src.prop4 import sweep_n_p_vs_n_fe, welfare_ratio_p_over_fe, figure_3_1_curves


def test_n_p_vs_n_fe_boundary_matches_analytic_condition():
    alphas = [0.05, 0.1, 0.15, 0.2, 0.25]
    sigmas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = sweep_n_p_vs_n_fe(alphas, sigmas)
    assert len(results) > 0
    for alpha, sigma, numeric, analytic in results:
        assert numeric == analytic, f"mismatch at alpha={alpha}, sigma={sigma}"


def test_welfare_ratio_diverges_as_alpha_sigma_to_zero_with_fixed_k():
    # paper's proof requires 0 < k < 1/3 for the limit (alpha, sigma) -> 0
    # with alpha/sigma -> k to satisfy stability condition (5.5)
    k = 0.2
    xs = [0.05, 0.01, 0.002, 0.0005]
    ratios = []
    for x in xs:
        sigma = x
        alpha = k * x
        ratios.append(welfare_ratio_p_over_fe(alpha, sigma, require_valid=False))
    # ratio should be increasing (diverging) as x shrinks
    assert ratios[-1] > ratios[0]


def test_figure_3_1_curves_shape():
    m = example1(alpha=0.1, sigma=0.3, c=1.0)
    n_grid = np.linspace(0.5, 20, 30)
    curves = figure_3_1_curves(m, n_grid, epsilon_F=1.0, epsilon_H=float("inf"))
    assert len(curves["proprietary_lhs"]) == len(n_grid)
    assert len(curves["open_lhs"]) == len(n_grid)
    assert len(curves["social_lhs"]) == len(n_grid)
    assert all(v >= 0 for v in curves["H_inv"])
