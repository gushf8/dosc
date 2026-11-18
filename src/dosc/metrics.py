"""
dosc.metrics
============
Diagnostic loss metrics and statistical equivalence testing for DOSC:
- Two One-Sided Test (TOST) of Schuirmann for parameter recovery equivalence
- Marginal Wasserstein-1 distributional distance
- Frobenius covariance structure loss

Author: Gustavo Fernando Huaman Fernandez
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from scipy.stats import t as student_t


@dataclass
class TOSTResult:
    """Result of Schuirmann's Two One-Sided Test for Equivalence."""
    delta: float
    diff: float
    se_diff: float
    df: int
    t_lower: float
    p_lower: float
    t_upper: float
    p_upper: float
    p_tost: float
    is_equivalent: bool
    minimum_delta: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Schuirmann Two One-Sided Test (TOST) for Statistical Equivalence",
            "=" * 60,
            f"Difference (Estimate - Benchmark): {self.diff:+.5f}",
            f"Standard Error of Difference:      {self.se_diff:.5f}",
            f"Degrees of freedom:                {self.df}",
            f"Equivalence margin (Delta):        +/- {self.delta:.5f}",
            f"Minimum feasible margin (Delta_min): +/- {self.minimum_delta:.5f}",
            f"t_lower: {self.t_lower:8.3f} (p = {self.p_lower:.4f})",
            f"t_upper: {self.t_upper:8.3f} (p = {self.p_upper:.4f})",
            f"TOST p-value: {self.p_tost:.4f}",
            f"Verdict: {'EQUIVALENT (H0 REJECTED at alpha=0.05)' if self.is_equivalent else 'NOT EQUIVALENT'}",
            "=" * 60
        ]
        return "\n".join(lines)


def two_one_sided_tost(
    beta_est: float,
    beta_ref: float,
    se_est: float,
    se_ref: float = 0.0,
    df: int = 20,
    delta: Optional[float] = None,
    alpha: float = 0.05
) -> TOSTResult:
    """Perform Schuirmann's Two One-Sided Test (TOST) for Statistical Equivalence.

    Tests whether the parameter estimate beta_est falls within a tolerance band [-Delta, +Delta]
    around the reference benchmark beta_ref:
        H0: |beta_est - beta_ref| >= Delta
        H1: |beta_est - beta_ref| < Delta

    Parameters
    ----------
    beta_est : float
        Synthetic cohort estimate (e.g., DOSC or Deaton).
    beta_ref : float
        Benchmark reference parameter (e.g., microdata or genuine panel).
    se_est : float
        Standard error of beta_est.
    se_ref : float, default=0.0
        Standard error of beta_ref. If 0.0, reference is treated as known ground truth.
    df : int, default=20
        Degrees of freedom for Student's t distribution.
    delta : float, optional
        Equivalence margin. If None, default is set to 2.0 * se_diff or 0.10 * |beta_ref|.
    alpha : float, default=0.05
        Significance level.
    """
    diff = float(beta_est - beta_ref)
    se_diff = float(np.sqrt(se_est ** 2 + se_ref ** 2))
    t_crit = float(student_t.ppf(1.0 - alpha, df=df))
    delta_min = float(t_crit * se_diff)

    if delta is None:
        delta = max(1.10 * delta_min, 0.10 * abs(beta_ref) if abs(beta_ref) > 1e-4 else 0.10)

    # Schuirmann's two one-sided test statistics
    # H01: beta_est - beta_ref <= -delta
    t_lower = (diff - (-delta)) / se_diff
    p_lower = 1.0 - float(student_t.cdf(t_lower, df=df))

    # H02: beta_est - beta_ref >= +delta
    t_upper = (delta - diff) / se_diff
    p_upper = 1.0 - float(student_t.cdf(t_upper, df=df))

    p_tost = max(p_lower, p_upper)
    is_equivalent = bool(p_tost < alpha)

    return TOSTResult(
        delta=delta,
        diff=diff,
        se_diff=se_diff,
        df=df,
        t_lower=t_lower,
        p_lower=p_lower,
        t_upper=t_upper,
        p_upper=p_upper,
        p_tost=p_tost,
        is_equivalent=is_equivalent,
        minimum_delta=delta_min
    )

