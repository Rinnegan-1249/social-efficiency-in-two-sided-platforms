"""Total social welfare W(theta_m, n), eq. (2.7), under constant-elasticity
F(theta) = B*theta**epsilon_F and H^{-1}(n) = C*n**(1/epsilon_H).

Closed forms for the two integral terms (derived by substitution, see
writeup.md):
    integral_0^theta_m theta*f(theta) d theta = epsilon_F/(1+epsilon_F) * theta_m * F(theta_m)
    integral_0^{H^{-1}(n)} phi*h(phi) d phi   = epsilon_H/(1+epsilon_H) * n * H^{-1}(n)
"""
from .equilibria import make_F, make_H_inv
from .model import PlatformModel


def welfare(model: PlatformModel, n: float, theta_m: float,
            epsilon_F: float, epsilon_H: float,
            B: float = 1.0, C: float = 1.0) -> float:
    F = make_F(B, epsilon_F)
    H_inv = make_H_inv(C, epsilon_H)

    user_cost_term = epsilon_F / (1 + epsilon_F) * theta_m * F(theta_m)
    if epsilon_H == float("inf"):
        dev_cost_term = n * H_inv(n)
    else:
        dev_cost_term = epsilon_H / (1 + epsilon_H) * n * H_inv(n)

    return model.V(n) * F(theta_m) - user_cost_term - dev_cost_term


def welfare_ratio(model: PlatformModel, n_num: float, theta_m_num: float,
                   n_den: float, theta_m_den: float,
                   epsilon_F: float, epsilon_H: float,
                   B: float = 1.0, C: float = 1.0) -> float:
    """W(n_num, theta_m_num) / W(n_den, theta_m_den)."""
    w_num = welfare(model, n_num, theta_m_num, epsilon_F, epsilon_H, B, C)
    w_den = welfare(model, n_den, theta_m_den, epsilon_F, epsilon_H, B, C)
    return w_num / w_den
