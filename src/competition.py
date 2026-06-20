"""Section 3.3 / Proposition 6: monopoly platform vs. two Hotelling-competing
platforms (paper pp. 26-30, 41-42).

Setup: users uniformly distributed on a unit Hotelling segment with
transport cost t; developers are platform-indifferent (perfect
substitutes) and multihome; all developers share the same fixed cost
phi (inelastic developer supply); within-platform competition follows
Example 2 (V(n) = A*n**beta).

Closed forms (paper p. 41):
    n_c = (beta*A / (2*phi)) ** (1/(1-beta))   -- developers per platform, duopoly
    n_p = (beta*A / phi)     ** (1/(1-beta))   -- developers, monopoly

Welfare gain from a single monopoly platform vs. two competing platforms:
    Delta_W(t) = A*(1-beta)*(n_p**beta - n_c**beta) - t/4

Proposition 6: there is a non-empty interval [t_L, t_H] such that total
welfare is higher under monopoly iff t in [t_L, t_H], where t_H solves
Delta_W(t_H) = 0 and t_L is the largest t consistent with both competing
platforms making non-negative profits while users singlehome.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionParams:
    A: float
    beta: float
    phi: float


def n_competing(p: CompetitionParams) -> float:
    """Developers entering (and multihoming) per platform under duopoly, eq. (3.13)."""
    return (p.beta * p.A / (2 * p.phi)) ** (1 / (1 - p.beta))


def n_monopoly(p: CompetitionParams) -> float:
    """Developers entering under a single monopoly platform, eq. (3.14)."""
    return (p.beta * p.A / p.phi) ** (1 / (1 - p.beta))


def welfare_gain(t: float, p: CompetitionParams) -> float:
    """Social welfare gain from a single monopoly platform vs. two
    Hotelling-competing platforms, as a function of transport cost t."""
    n_p, n_c = n_monopoly(p), n_competing(p)
    return p.A * (1 - p.beta) * (n_p ** p.beta - n_c ** p.beta) - t / 4


def t_H(p: CompetitionParams) -> float:
    """Transport cost above which competition beats monopoly (welfare_gain(t_H) = 0)."""
    beta, phi, A = p.beta, p.phi, p.A
    base = (beta * A / (2 * phi)) ** (beta / (1 - beta))
    return 4 * A * (1 - beta) * base * (2 ** (beta / (1 - beta)) - 1)


def t_L(p: CompetitionParams) -> float:
    """Smallest transport cost for which the duopoly equilibrium is valid
    (non-negative platform profits, users singlehome)."""
    beta, phi, A = p.beta, p.phi, p.A
    n_c = n_competing(p)
    return 2 * beta * A * n_c ** beta


def find_valid_interval(p: CompetitionParams):
    """Return (t_L, t_H); non-empty (t_L <= t_H) for any beta >= 0 (paper proof)."""
    return t_L(p), t_H(p)
