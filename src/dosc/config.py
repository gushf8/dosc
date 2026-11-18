"""
dosc.config
===========
Hyperparameters, feasibility thresholds, and optimization configuration for DOSC.

Author: Gustavo Fernando Huaman Fernandez
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DOSCConfig:
    """Configuration dataclass for Dynamic Optimal Synthetic Cohorts (DOSC).

    Parameters
    ----------
    tau_d : float, default=0.55
        Maximum admissible Wasserstein-1 distributional divergence tolerance (D_dist).
    tau_c : float, default=0.25
        Maximum admissible Frobenius covariance dependency loss tolerance (D_dep).
    tau_eiv : float, default=0.35
        Maximum noise-to-signal ratio tolerance for errors-in-variables (D_eiv).
        Recommended regime: 0.25 - 0.50 for continuous regressors in simulated DGPs;
        3.0 - 5.0 for real household survey microdata with discrete/count treatments.
    tau_i : float, default=0.35
        Backward-compatible alias for tau_eiv (strictly unsupervised, independent of Y).
    tau_s : float, default=0.85
        Minimum admissible longitudinal silhouette temporal stability threshold (S_temp).
    n_min : float, default=15.0
        Minimum Kish effective sample size per synthetic cohort cell (n_eff).
    G_min : int, default=4
        Minimum candidate number of synthetic cohorts to evaluate.
    G_max : int, default=60
        Maximum candidate number of synthetic cohorts to evaluate.
    step_size : int, default=2
        Grid search step size for candidate G values.
    soft_assignment : bool, default=True
        Whether to compute continuous soft-assignment weights via temperature-scaled Mahalanobis distance.
    gamma_temperature : float, default=1.0
        Softmax temperature parameter gamma > 0 controlling membership dispersion.
    use_dim_reduction : bool, default=False
        Whether to project high-dimensional invariant anchors onto principal components.
    target_dim : int, default=6
        Target number of principal components when dimension reduction is active.
    pca_variance_threshold : float, default=0.85
        Cumulative explained variance ratio required when retaining principal components.
    cov_epsilon : float, default=1e-4
        Numerical regularization constant added to covariance diagonal for Mahalanobis stability.
    complexity_penalty : float, default=0.05
        Parsimonious cardinality penalty coefficient Pen(G) = complexity_penalty * log(G).
    early_stop_on_first_feasible : bool, default=False
        Whether to terminate grid search upon encountering the first feasible candidate G.
    """

    tau_d: float = 0.55
    tau_c: float = 0.25
    tau_eiv: float = 0.35
    tau_i: float = 0.35
    tau_s: float = 0.85
    n_min: float = 15.0
    G_min: int = 4
    G_max: int = 60
    step_size: int = 2

    soft_assignment: bool = True
    gamma_temperature: float = 1.0

    use_dim_reduction: bool = False
    target_dim: int = 6
    pca_variance_threshold: float = 0.85

    cov_epsilon: float = 1e-4
    complexity_penalty: float = 0.05
    early_stop_on_first_feasible: bool = False

