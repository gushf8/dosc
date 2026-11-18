"""
DOSC: Dynamic Optimal Synthetic Cohorts
=======================================
A data-driven synthetic cohort estimator with endogenous dimension selection,
multivariate invariant anchors, and finite-sample measurement error correction
for repeated cross-sectional microdata.

Author: Gustavo Fernando Huaman Fernandez
Affiliation: Escuela de Posgrado, Universidad Tecnológica del Perú (UTP)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Gustavo Fernando Huaman Fernandez"

from .config import DOSCConfig
from .engine import DOSCEngine, PrototypeCatalog, PseudoPanelResult
from .estimators import (
    SyntheticCohortModel,
    fit_synthetic_wls,
    fit_deaton_cohorts
)
from .metrics import (
    TOSTResult,
    two_one_sided_tost
)
from .dgp import generate_synthetic_panel_dgp

__all__ = [
    "DOSCConfig",
    "DOSCEngine",
    "PrototypeCatalog",
    "PseudoPanelResult",
    "SyntheticCohortModel",
    "fit_synthetic_wls",
    "fit_deaton_cohorts",
    "TOSTResult",
    "two_one_sided_tost",
    "generate_synthetic_panel_dgp",
    "__version__",
    "__author__"
]

