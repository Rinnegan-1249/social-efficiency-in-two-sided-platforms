"""Section 3.2 / Proposition 5: vertically differentiated developers.

Developers are exogenously differentiated by application quality q in
[qL, qH], uniformly costed (H(q) = C*(q - qL), so h(q) = C), and the
platform can only charge fixed access fees. Gross user surplus from
admitting developers with quality >= qm is V_bar(Q(qm)) = Vbar - A/Q**beta,
where Q(qm) = (C/2)*(qH**2 - qm**2) is the expected quality-weighted
"quantity" bought by a user (paper pp. 23-25).

Proposition 5 (paper p. 26): if beta*qL > 2*qH, then Pi_P(qm) and W(qm)
are both concave in qm, and the profit-maximizing marginal quality
qm_p is *below* the socially optimal qm_so -- i.e. the proprietary
platform induces *excessive* product variety (more developers admitted
than is socially optimal), contrary to the horizontal-differentiation
case (Proposition 4).
"""
from dataclasses import dataclass

from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class VerticalDiffParams:
    qL: float
    qH: float
    C: float       # density of the (uniform) quality distribution, h(q) = C
    B: float       # mass of (identical, inelastic) users
    Vbar: float    # V(Q) = Vbar - A / Q**beta
    A: float
    beta: float
    phi: float     # common developer fixed cost


def sufficient_condition_holds(p: VerticalDiffParams) -> bool:
    """beta*qL > 2*qH, Proposition 5's sufficient condition."""
    return p.beta * p.qL > 2 * p.qH


def Q(qm: float, p: VerticalDiffParams) -> float:
    return (p.C / 2) * (p.qH ** 2 - qm ** 2)


def V_of_Q(Qval: float, p: VerticalDiffParams) -> float:
    return p.Vbar - p.A / Qval ** p.beta


def V_prime_of_Q(Qval: float, p: VerticalDiffParams) -> float:
    return p.A * p.beta * Qval ** (-p.beta - 1)


def E(qm: float, p: VerticalDiffParams) -> float:
    """Uninternalized developer surplus per user, E(qm) (paper p. 25)."""
    Qval = Q(qm, p)
    return p.C * V_prime_of_Q(Qval, p) / 2 * (p.qH - qm) ** 2


def profit(qm: float, p: VerticalDiffParams) -> float:
    """Platform profit Pi_P(qm) (paper p. 25-26)."""
    return p.B * (V_of_Q(Q(qm, p), p) - E(qm, p)) - p.C * (p.qH - qm) * p.phi


def welfare(qm: float, p: VerticalDiffParams) -> float:
    """Total social welfare W(qm) (paper p. 26)."""
    return p.B * V_of_Q(Q(qm, p), p) - p.C * (p.qH - qm) * p.phi


def _maximize(f, p: VerticalDiffParams, eps: float = 1e-9) -> float:
    """Maximize f(qm, p) over the open interval (qL, qH)."""
    res = minimize_scalar(lambda qm: -f(qm, p), bounds=(p.qL + eps, p.qH - eps),
                           method="bounded", options={"xatol": 1e-12})
    return res.x


def solve_qm_p(p: VerticalDiffParams) -> float:
    return _maximize(profit, p)


def solve_qm_so(p: VerticalDiffParams) -> float:
    return _maximize(welfare, p)
