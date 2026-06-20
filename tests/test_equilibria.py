import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import example1, example2
from src.equilibria import solve_equilibria, check_assumptions


def test_proposition4_i_insufficient_variety_example1():
    """Both proprietary and open platforms under-provide variety/adoption
    relative to the social optimum (Proposition 4.i)."""
    m = example1(alpha=0.2, sigma=0.4, c=1.0)
    res = solve_equilibria(m, epsilon_F=1.0, epsilon_H=2.0, B=1.0, C=1.0)
    assert res.n_p < res.n_so
    assert res.n_fe < res.n_so
    assert res.theta_m_p < res.theta_m_so
    assert res.theta_m_fe < res.theta_m_so


def test_proposition4_i_insufficient_variety_example2():
    m = example2(A=2.0, beta=0.5)
    res = solve_equilibria(m, epsilon_F=1.0, epsilon_H=2.0, B=1.0, C=1.0)
    assert res.n_p < res.n_so
    assert res.n_fe < res.n_so


def test_invalid_parameters_raise_by_default():
    m = example1(alpha=0.45, sigma=0.49, c=1.0)
    assert not check_assumptions(m.epsilon_V, m.lambda_ratio, epsilon_F=20, epsilon_H=20)
    try:
        solve_equilibria(m, epsilon_F=20, epsilon_H=20)
        assert False, "expected ValueError"
    except ValueError:
        pass
