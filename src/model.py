"""Closed-form developer/user surplus primitives from Hagiu (2007), pp. 7-10.

Two parameterized examples are provided, both satisfying Assumption 1
(u increasing, pi decreasing, V increasing & concave):

- Example 1: Spence-Dixit-Stiglitz demand, parameters (alpha, sigma, c),
  0 < alpha < sigma < 1.
- Example 2: unit demand, V(n) = A * n**beta, 0 < beta < 1.

Each example exposes V(n), V'(n), pi(n), u(n), and the constants
epsilon_V (elasticity of V) and lambda_ratio = pi(n) / V'(n).
"""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PlatformModel:
    """Bundle of model primitives for a given parameterization."""
    V: Callable[[float], float]
    V_prime: Callable[[float], float]
    pi: Callable[[float], float]
    u: Callable[[float], float]
    epsilon_V: float
    lambda_ratio: float


def example1(alpha: float, sigma: float, c: float) -> PlatformModel:
    """Spence-Dixit-Stiglitz example (paper p. 8).

    Requires 0 < alpha < sigma < 1, c > 0.
    """
    if not (0 < alpha < sigma < 1):
        raise ValueError("require 0 < alpha < sigma < 1")
    if c <= 0:
        raise ValueError("c must be positive")

    k = (alpha * sigma / c) ** (alpha / (1 - alpha))
    epsilon_V = alpha * (1 - sigma) / (sigma * (1 - alpha))
    lambda_ratio = sigma * (1 - alpha) / (1 - sigma * alpha)
    exponent = alpha * (1 - sigma) / (sigma * (1 - alpha))  # == epsilon_V

    def V(n: float) -> float:
        return (1 - sigma * alpha) * k * n ** exponent

    def V_prime(n: float) -> float:
        return (1 - sigma * alpha) * k * exponent * n ** (exponent - 1)

    def pi(n: float) -> float:
        return (1 - sigma) * alpha * k * n ** (-(sigma - alpha) / (sigma * (1 - alpha)))

    def u(n: float) -> float:
        return (1 - alpha) * k * n ** exponent

    return PlatformModel(V=V, V_prime=V_prime, pi=pi, u=u,
                          epsilon_V=epsilon_V, lambda_ratio=lambda_ratio)


def example2(A: float, beta: float) -> PlatformModel:
    """Unit-demand example (paper p. 9-10): V(n) = A * n**beta, 0 < beta < 1."""
    if not (0 < beta < 1):
        raise ValueError("require 0 < beta < 1")
    if A <= 0:
        raise ValueError("A must be positive")

    def V(n: float) -> float:
        return A * n ** beta

    def V_prime(n: float) -> float:
        return A * beta * n ** (beta - 1)

    def pi(n: float) -> float:
        return A * beta * n ** (beta - 1)

    def u(n: float) -> float:
        return (1 - beta) * A * n ** beta

    return PlatformModel(V=V, V_prime=V_prime, pi=pi, u=u,
                          epsilon_V=beta, lambda_ratio=1.0)
