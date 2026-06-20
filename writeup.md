# Writeup: Numerical Comparative Statics for Hagiu (2007)

Source: Andrei Hagiu, "Proprietary vs. Open Two-Sided Platforms and Social
Efficiency," HBS Working Paper 07-095 (2006/2007).

## What the paper claims

Hagiu models a two-sided platform connecting users and developers, where the
platform's value to each side is increasing in participation on the other
side (positive indirect network effects). It compares three regimes:

- **Proprietary**: a profit-maximizing platform charges prices `P^U`, `P^D`
  on both sides.
- **Open / free entry**: prices are zero on both sides; users and developers
  enter until surplus/profit opportunities are exhausted.
- **Social planner**: chooses entry on both sides to maximize total welfare.

The paper's central, counterintuitive result: because a proprietary platform
sets prices to maximize profit, it partially *internalizes* the positive
indirect network effects and the direct (business-stealing) competitive
effects between developers — effects that an open platform, pricing at
marginal cost, leaves completely uninternalized. This can make the
proprietary platform's product variety, user adoption, and total welfare
*higher* than the open platform's, even though the proprietary platform also
creates monopoly deadweight loss. The same logic implies that platform
*competition* can be socially undesirable, and that with vertically
differentiated developers a proprietary platform can induce *excessive*
(not just insufficient) variety.

## What was implemented

| Module | Content |
|---|---|
| `src/model.py` | Closed-form `V(n)`, `V'(n)`, `pi(n)`, `u(n)` for Example 1 (Spence-Dixit-Stiglitz) and Example 2 (unit demand), paper pp. 7-10. |
| `src/equilibria.py` | Root-finds eqs. (3.4)/(3.5)/(3.6) for `n_p`, `n_fe`, `n_so` under constant-elasticity `F(theta)` and `H^{-1}(n)`; implements Lemma 0's well-definedness guards (eqs. 2.12/2.13). |
| `src/welfare.py` | Closed-form total welfare `W(n, theta_m)`, eq. (2.7), derived by direct integration of the constant-elasticity demand curves. |
| `src/prop4.py` | Reproduces Proposition 4: the `n_p` vs. `n_fe` boundary condition, the two welfare-ratio limits, and Figure 3.1 (graphs a/b). |
| `src/vertical_diff.py` | Reproduces Section 3.2 / Proposition 5: vertically differentiated developers, `Q(qm)`, `E(qm)`, and the excessive-variety result. |
| `src/competition.py` | Reproduces Section 3.3 / Proposition 6: monopoly vs. Hotelling-competing platforms, and the `[t_L, t_H]` interval where monopoly is welfare-superior. |

All headline numeric claims are covered by `pytest` unit tests
(`tests/`, 22 tests) *and* re-verified live inside `notebooks/results.ipynb`.

## What each figure shows

- **`prop4_boundary.png`** — sweeps `(alpha, sigma)` and numerically solves for
  `n_p`, `n_fe` under Prop 4.ii's restriction (`epsilon_F=1`, inelastic
  developer supply). Colors which region has `n_p > n_fe`; this is checked
  point-by-point against the paper's closed-form boundary
  `(1-sigma*alpha)^2 > 2*sigma*(1-alpha)^2`. Over 975 valid grid points, 0
  mismatches.
- **`prop4_welfare_ratio_limits.png`** — left panel: `W(n_p)/W(n_fe)` as
  `alpha, sigma -> 0` with `alpha/sigma -> k` fixed; diverges, matching the
  paper. (Note: the paper's proof requires `k < 1/3` for the relevant
  stability condition (5.5) to hold in the limit — using `k >= 1/3`
  numerically produces equilibria that violate Lemma 0 and is *not* a
  counterexample to the paper, just outside its stated regime.) Right panel:
  ratio as `sigma -> 1` (alpha fixed); converges to 3/4 as claimed.
- **`prop4_figure_3_1.png`** — recreates the paper's Figure 3.1: the LHS
  curves of (3.4)/(3.5)/(3.6) plotted against the RHS entry-cost curves, for
  a parameterization where business-stealing dominates (graph a, decreasing
  curves) and one where indirect network effects dominate (graph b,
  increasing curves). Vertical lines mark the resulting `n_p`, `n_fe`, `n_so`.
- **`prop5_excessive_variety.png`** — `Pi_P(qm)` and `W(qm)` for vertically
  differentiated developers under a parameterization satisfying
  `beta*qL > 2*qH`. Both are concave, but the profit-maximizing threshold
  `qm_p` is *below* the socially optimal `qm_so` — the proprietary platform
  admits more (lower-quality) developers than is socially optimal, reversing
  the usual under-provision result from Proposition 4.
- **`prop6_interval.png`** — welfare gain `W(monopoly) - W(duopoly)` as a
  function of Hotelling transport cost `t`; shows the non-empty interval
  `[t_L, t_H]` (here `[3.33, 6.67]`) over which a single monopoly platform is
  welfare-superior to two competing platforms, because competition prevents
  platforms from internalizing positive indirect network effects enough to
  induce sufficient developer entry.

## Numerical edge cases encountered

- **Degenerate exponent at `epsilon_V = 0.5` exactly** (with `epsilon_F=1`):
  the LHS of eqs. (3.4)/(3.5) becomes an exact power of `n^0`, i.e. flat —
  no root exists. This occurs exactly at the equality boundary of Lemma 0's
  condition (2.12). `check_assumptions` uses a small numerical margin (not a
  strict `<`, which floating-point arithmetic can spuriously satisfy at the
  exact boundary) to exclude these points.
- **Root bracketing at extreme parameter values**: for some `(alpha, sigma)`
  combinations the equilibrium `n` can be extremely small (`< 1e-9`).
  `equilibria._find_root` expands the search bracket in *both* directions
  (shrinking `lo`, growing `hi`) rather than only growing `hi`, since a purely
  one-sided expansion failed to bracket some of these roots.
- **Proposition 4.ii's `k < 1/3` restriction**: the paper's proof of the
  `alpha, sigma -> 0` divergence limit explicitly requires `0 < k < 1/3` (not
  `k < 1`, as a naive reading of "alpha/sigma -> k < 1" earlier in the same
  proposition might suggest) for the stability condition (5.5) to hold in the
  limit. The notebook and tests use `k=0.2`, consistent with this restriction.
- **Proposition 5 parameterization**: the sufficient condition
  `beta*qL > 2*qH` alone does not guarantee an *interior* optimum — if `A`
  (the `V(Q) = Vbar - A/Q**beta` scale parameter) is too large relative to
  the developer fixed cost `phi`, both `Pi_P(qm)` and `W(qm)` are maximized
  at the boundary `qm = qL` (admit everyone), which trivially satisfies
  `qm_p <= qm_so` without exercising the proposition's actual mechanism. The
  chosen parameters (`qL=1, qH=2, beta=5, A=1, phi=1`) were picked so that
  `phi > B*qL*V'(Q(qL))`, guaranteeing both optima are interior.
