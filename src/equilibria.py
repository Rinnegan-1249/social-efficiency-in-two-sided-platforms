"""Solve for product variety n and marginal user theta_m under the three
regimes analyzed in Hagiu (2007) Section 2-3: proprietary platform (eqs.
2.5-2.6, reduced to eq. 3.4), open/free-entry platform (eq. 3.5), and the
social planner (eq. 3.6).

User demand elasticity F(theta) = B * theta**epsilon_F and the developer
entry-cost curve H^{-1}(n) = C * n**(1/epsilon_H) are assumed to have
constant elasticity, matching Lemma 0 in the paper. epsilon_H may be set
to float("inf") to represent perfectly inelastic developer supply (all
developers share the same fixed cost).
"""
from dataclasses import dataclass
from typing import Callable, NamedTuple

from scipy.optimize import brentq

from .model import PlatformModel


def make_F(B: float, epsilon_F: float) -> Callable[[float], float]:
    return lambda theta: B * theta ** epsilon_F


def make_H_inv(C: float, epsilon_H: float) -> Callable[[float], float]:
    if epsilon_H == float("inf"):
        return lambda n: C
    return lambda n: C * n ** (1.0 / epsilon_H)


def make_H_inv_prime(C: float, epsilon_H: float) -> Callable[[float], float]:
    if epsilon_H == float("inf"):
        return lambda n: 0.0
    return lambda n: (C / epsilon_H) * n ** (1.0 / epsilon_H - 1.0)


def check_assumptions(epsilon_V: float, lambda_ratio: float,
                       epsilon_F: float, epsilon_H: float) -> bool:
    """Lemma 0 well-definedness / stability guards, eqs. (2.12) and (2.13)."""
    if not (0 < epsilon_V < 1):
        return False
    inv_epsilon_H = 0.0 if epsilon_H == float("inf") else 1.0 / epsilon_H

    # margin avoids the boundary case epsilon_V*(1+epsilon_F) == 1 + 1/epsilon_H, where the
    # LHS of (3.4)/(3.5)/(3.6) becomes exactly flat in n (degenerate, no/non-unique root)
    margin = 1e-6
    cond_212 = epsilon_V * (1 + epsilon_F) < 1 + inv_epsilon_H - margin

    bracket = (1 - lambda_ratio * epsilon_V) / (1 - epsilon_V) * epsilon_V * (1 + epsilon_F) - 1
    cond_213 = bracket * lambda_ratio * (1 - epsilon_V) < 1.0 / (1 + (0.0 if epsilon_H == float("inf") else epsilon_H)) - margin

    return cond_212 and cond_213


def _find_root(g: Callable[[float], float], lo: float = 1e-9, hi: float = 1e6) -> float:
    """Bracket and solve g(n) = 0 for n > 0, expanding lo/hi outward if needed."""
    f_lo = g(lo)
    f_hi = g(hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 80:
        if tries % 2 == 0:
            hi *= 2
            f_hi = g(hi)
        else:
            lo /= 2
            f_lo = g(lo)
        tries += 1
    if f_lo * f_hi > 0:
        raise RuntimeError("could not bracket a root; check parameters / assumptions")
    return brentq(g, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=300)


class EquilibriumResult(NamedTuple):
    n_p: float
    n_fe: float
    n_so: float
    theta_m_p: float
    theta_m_fe: float
    theta_m_so: float


def solve_equilibria(model: PlatformModel, epsilon_F: float, epsilon_H: float,
                      B: float = 1.0, C: float = 1.0,
                      require_valid: bool = True) -> EquilibriumResult:
    """Solve eqs. (3.4) proprietary, (3.5) open, (3.6) social-planner for n,
    and the corresponding marginal user theta_m, given a PlatformModel
    (V, V', pi, u) and constant-elasticity F, H^{-1}.
    """
    if require_valid and not check_assumptions(model.epsilon_V, model.lambda_ratio,
                                                epsilon_F, epsilon_H):
        raise ValueError("parameters violate Lemma 0 assumptions (2.12)/(2.13); "
                          "pass require_valid=False to bypass")

    F = make_F(B, epsilon_F)
    H_inv = make_H_inv(C, epsilon_H)
    H_inv_prime = make_H_inv_prime(C, epsilon_H)

    ratio = epsilon_F / (1 + epsilon_F)

    def g_p(n: float) -> float:
        theta = ratio * model.V(n)
        return model.V_prime(n) * F(theta) - (n * H_inv_prime(n) + H_inv(n))

    def g_fe(n: float) -> float:
        return model.pi(n) * F(model.u(n)) - H_inv(n)

    def g_so(n: float) -> float:
        return model.V_prime(n) * F(model.V(n)) - H_inv(n)

    n_p = _find_root(g_p)
    n_fe = _find_root(g_fe)
    n_so = _find_root(g_so)

    theta_m_p = ratio * model.V(n_p)
    theta_m_fe = model.u(n_fe)
    theta_m_so = model.V(n_so)

    return EquilibriumResult(n_p=n_p, n_fe=n_fe, n_so=n_so,
                              theta_m_p=theta_m_p, theta_m_fe=theta_m_fe,
                              theta_m_so=theta_m_so)
