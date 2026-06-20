"""Helpers to reproduce Proposition 4 (paper Section 3.1, pp. 19-21).

Proposition 4(i): both n_p and n_fe are below the socially optimal n_so
(reproduced/tested directly via equilibria.solve_equilibria, see
tests/test_equilibria.py).

Proposition 4(ii) restricts to epsilon_F = 1 and inelastic developer
supply (epsilon_H = inf, all developers share fixed cost phi) under
Example 1. In that special case:
    n_p > n_fe  iff  (1 - sigma*alpha)**2 > 2*sigma*(1 - alpha)**2
and the welfare ratio W(n_p)/W(n_fe):
    -> +inf as alpha,sigma -> 0 with alpha/sigma -> k < 1
    -> 3/4  as sigma -> 1 (alpha fixed)
"""
from typing import List, Tuple

from .equilibria import check_assumptions, solve_equilibria, make_F, make_H_inv
from .model import PlatformModel, example1
from .welfare import welfare

EPSILON_F_PROP4_II = 1.0
EPSILON_H_PROP4_II = float("inf")


def boundary_condition(alpha: float, sigma: float) -> bool:
    """Analytic n_p > n_fe boundary from Proposition 4.ii."""
    return (1 - sigma * alpha) ** 2 > 2 * sigma * (1 - alpha) ** 2


def sweep_n_p_vs_n_fe(alphas: List[float], sigmas: List[float],
                       c: float = 1.0, B: float = 1.0, C: float = 1.0
                       ) -> List[Tuple[float, float, bool, bool]]:
    """For each valid (alpha, sigma) with alpha < sigma, solve numerically
    for n_p, n_fe under the Prop 4.ii restriction (epsilon_F=1, epsilon_H=inf)
    and compare the sign of n_p - n_fe to the analytic boundary condition.

    Returns list of (alpha, sigma, n_p_greater_numeric, n_p_greater_analytic).
    """
    results = []
    for alpha in alphas:
        for sigma in sigmas:
            if not (0 < alpha < sigma < 1):
                continue
            model = example1(alpha=alpha, sigma=sigma, c=c)
            if not check_assumptions(model.epsilon_V, model.lambda_ratio,
                                      EPSILON_F_PROP4_II, EPSILON_H_PROP4_II):
                continue
            try:
                res = solve_equilibria(model, epsilon_F=EPSILON_F_PROP4_II,
                                        epsilon_H=EPSILON_H_PROP4_II, B=B, C=C)
            except RuntimeError:
                # numerically degenerate point (e.g. epsilon_V very close to 0.5,
                # which makes the LHS of (3.4)/(3.5) nearly flat in n); skip it
                continue
            numeric = res.n_p > res.n_fe
            analytic = boundary_condition(alpha, sigma)
            results.append((alpha, sigma, numeric, analytic))
    return results


def welfare_ratio_p_over_fe(alpha: float, sigma: float, c: float = 1.0,
                             B: float = 1.0, C: float = 1.0,
                             require_valid: bool = True) -> float:
    model = example1(alpha=alpha, sigma=sigma, c=c)
    res = solve_equilibria(model, epsilon_F=EPSILON_F_PROP4_II,
                            epsilon_H=EPSILON_H_PROP4_II, B=B, C=C,
                            require_valid=require_valid)
    w_p = welfare(model, res.n_p, res.theta_m_p, EPSILON_F_PROP4_II, EPSILON_H_PROP4_II, B, C)
    w_fe = welfare(model, res.n_fe, res.theta_m_fe, EPSILON_F_PROP4_II, EPSILON_H_PROP4_II, B, C)
    return w_p / w_fe


def figure_3_1_curves(model: PlatformModel, n_grid, epsilon_F: float, epsilon_H: float,
                       B: float = 1.0, C: float = 1.0) -> dict:
    """LHS of eqs. (3.4) proprietary, (3.5) open, (3.6) social planner, plus
    the two RHS entry-cost curves H^{-1}(n) and n*H^{-1'}(n)+H^{-1}(n), each
    evaluated over n_grid -- used to recreate paper Figure 3.1 (graphs a/b).
    """
    F = make_F(B, epsilon_F)
    H_inv = make_H_inv(C, epsilon_H)
    ratio = epsilon_F / (1 + epsilon_F)

    if epsilon_H == float("inf"):
        H_inv_prime = lambda n: 0.0
    else:
        H_inv_prime = lambda n: (C / epsilon_H) * n ** (1.0 / epsilon_H - 1.0)

    proprietary_lhs = [model.V_prime(n) * F(ratio * model.V(n)) for n in n_grid]
    open_lhs = [model.pi(n) * F(model.u(n)) for n in n_grid]
    social_lhs = [model.V_prime(n) * F(model.V(n)) for n in n_grid]
    h_inv_curve = [H_inv(n) for n in n_grid]
    h_inv_plus_marginal = [n * H_inv_prime(n) + H_inv(n) for n in n_grid]

    return {
        "n": list(n_grid),
        "proprietary_lhs": proprietary_lhs,
        "open_lhs": open_lhs,
        "social_lhs": social_lhs,
        "H_inv": h_inv_curve,
        "nH_inv_prime_plus_H_inv": h_inv_plus_marginal,
    }
