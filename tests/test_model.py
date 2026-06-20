import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import example1, example2


def test_example1_epsilon_v_and_lambda():
    alpha, sigma, c = 0.3, 0.6, 1.0
    m = example1(alpha, sigma, c)
    expected_epsilon_v = alpha * (1 - sigma) / (sigma * (1 - alpha))
    expected_lambda = sigma * (1 - alpha) / (1 - sigma * alpha)
    assert math.isclose(m.epsilon_V, expected_epsilon_v)
    assert math.isclose(m.lambda_ratio, expected_lambda)
    assert 0 < m.epsilon_V < 1
    assert 0 < m.lambda_ratio < 1


def test_example1_V_equals_u_plus_n_pi():
    alpha, sigma, c = 0.2, 0.5, 2.0
    m = example1(alpha, sigma, c)
    for n in (1.0, 5.0, 50.0):
        assert math.isclose(m.V(n), m.u(n) + n * m.pi(n), rel_tol=1e-9)


def test_example1_V_prime_matches_numeric_derivative():
    alpha, sigma, c = 0.25, 0.55, 1.5
    m = example1(alpha, sigma, c)
    n0 = 10.0
    h = 1e-5
    numeric = (m.V(n0 + h) - m.V(n0 - h)) / (2 * h)
    assert math.isclose(m.V_prime(n0), numeric, rel_tol=1e-4)


def test_example1_requires_alpha_lt_sigma():
    try:
        example1(0.7, 0.5, 1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_example2_closed_forms():
    A, beta = 3.0, 0.4
    m = example2(A, beta)
    assert m.epsilon_V == beta
    assert m.lambda_ratio == 1.0
    n = 7.0
    assert math.isclose(m.pi(n), m.V_prime(n))
    assert math.isclose(m.u(n), m.V(n) - n * m.V_prime(n))


def test_example2_V_prime_matches_numeric_derivative():
    A, beta = 2.0, 0.6
    m = example2(A, beta)
    n0 = 8.0
    h = 1e-5
    numeric = (m.V(n0 + h) - m.V(n0 - h)) / (2 * h)
    assert math.isclose(m.V_prime(n0), numeric, rel_tol=1e-4)
