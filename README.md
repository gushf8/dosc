# dosc: Dynamic Optimal Synthetic Cohorts

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24326322.svg)](https://doi.org/10.5281/zenodo.24326322)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Author: Gustavo Fernando Huaman Fernandez](https://img.shields.io/badge/author-Gustavo%20Huaman%20Fernandez-purple.svg)]()

**dosc** is an open-source Python library for constructing **optimal pseudo-panels** from repeated cross-sectional survey microdata when true individual panel tracking is unavailable.

Developed and designed by **Gustavo Fernando Huaman Fernandez** (Escuela de Posgrado, Universidad Tecnológica del Perú).

---

## 📌 What is `dosc`?

In applied economics, health, and social sciences, researchers often analyze repeated cross-sections (e.g., household surveys, labor surveys). Traditional pseudo-panels (Deaton, 1985) group individuals into arbitrary 5-year birth brackets, causing:
1. Arbitrary cell cardinality with unoptimized bias-variance tradeoffs.
2. Hidden within-cohort heterogeneity and measurement error.
3. Severe attenuation bias (underestimating structural effects by 40%–80%).
4. Hard boundary collapse for borderline observations.

**DOSC solves this** by discovering latent invariant profiles $Z_i$, assigning continuous soft Mahalanobis weights $p_{igt}$, calculating Kish effective sample sizes $n_{gt}^{\text{eff}}$, and estimating Two-Way Fixed Effects models with analytical finite-sample Errors-in-Variables (EIV) corrections.

---

## 🚀 Quick Installation

Install directly from GitHub via `pip`:

```bash
pip install git+https://github.com/gushf8/dosc.git
```

Or clone and install locally:

```bash
git clone https://github.com/gushf8/dosc.git
cd dosc
pip install -e .
```

---

## ⚡ Quickstart: 5-Minute Minimal Example

```python
import pandas as pd
from dosc import DOSCEngine, DOSCConfig, fit_synthetic_wls

# 1. Load your repeated cross-sectional survey
# Required columns: invariant characteristics (Z), regressors (X), outcome (Y), time wave, sampling weights
df = pd.read_csv("your_repeated_survey.csv")

# 2. Configure the DOSC Engine
config = DOSCConfig(
    G_min=4,              # Minimum number of synthetic cohorts to explore
    G_max=24,             # Maximum number of synthetic cohorts to explore
    step_size=2,          # Search grid step
    gamma_temperature=1.0,# Temperature for soft Mahalanobis assignment
    tau_s=0.70,           # Minimum longitudinal stability threshold
    n_min=15.0            # Minimum Kish effective cell size
)

engine = DOSCEngine(config=config)

# 3. Fit DOSC on your DataFrame
res = engine.fit_dataframe(
    df=df,
    time_col="survey_year",
    z_cols=["birth_year", "gender", "stratum_code"], # Invariant anchors
    x_cols=["education_years", "experience"],       # Structural regressors
    y_col="log_wage",                               # Outcome
    weight_col="sampling_weight"                    # Expansion weight
)

if res.status == "SUCCESS":
    print(f"Optimal Cohorts Selected: G* = {res.G_star}")
    
    # 4. Estimate Two-Way Fixed Effects with Synthetic WLS & EIV Correction
    model = fit_synthetic_wls(
        pseudo_df=res.pseudo_panel_df,
        x_cols=["education_years", "experience"],
        y_col="Y_bar",
        cohort_col="cohort_g",
        time_col="time_t",
        eiv_correction=True
    )
    print(model.summary())
else:
    print("Rejection diagnostics:", res.rejection_reasons)
```

---

## ⚙️ Core Configuration Hyperparameters (`DOSCConfig`)

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `G_min` | `int` | `4` | Lower bound for candidate cohort cardinalities. |
| `G_max` | `int` | `24` | Upper bound for candidate cohort cardinalities. |
| `step_size` | `int` | `2` | Evaluation step across candidate cohort grid. |
| `gamma_temperature`| `float` | `1.0` | Softmax temperature scaling over Mahalanobis distance. |
| `tau_d` | `float` | `0.65` | Maximum admissible Wasserstein-1 distributional loss. |
| `tau_c` | `float` | `0.30` | Maximum admissible Frobenius covariance loss. |
| `tau_eiv` | `float` | `0.45` | Maximum admissible unsupervised noise-to-signal ratio. |
| `tau_s` | `float` | `0.70` | Minimum longitudinal silhouette stability threshold. |
| `n_min` | `float` | `15.0` | Minimum Kish effective cell size ($n_{\text{eff}} \ge 15.0$). |

---

## 🏛️ Authorship and Development

* **Author & Developer:** **Gustavo Fernando Huaman Fernandez**
* **Affiliation:** Escuela de Posgrado, Universidad Tecnológica del Perú (UTP)
* **ORCID:** [0009-0006-6570-2582](https://orcid.org/0009-0006-6570-2582)
* **Email:** E26203243@postgradoutp.edu.pe
* **Release:** November 2026

---

## 📖 Citation (Mandatory Condition of Use)

If you use **DOSC** in your academic papers, empirical research, software projects, or policy evaluations, please cite the methodological paper and Zenodo DOI as follows:

### APA 7th Edition
```text
HUAMAN FERNANDEZ, G. F. (2026). Can Repeated Cross-Sections Replicate Panels? Optimal Synthetic Cohorts for Complex Survey Econometrics. https://doi.org/10.5281/zenodo.24326322
```

### BibTeX
```bibtex
@article{huaman2026dosc,
  author       = {Huaman Fernandez, Gustavo Fernando},
  title        = {Can Repeated Cross-Sections Replicate Panels? Optimal Synthetic Cohorts for Complex Survey Econometrics},
  year         = {2026},
  doi          = {10.5281/zenodo.24326322},
  url          = {https://doi.org/10.5281/zenodo.24326322},
  journal      = {Zenodo Preprint}
}
```

### RIS (EndNote, Zotero, Mendeley)
```text
TY  - JOUR
AU  - Huaman Fernandez, Gustavo Fernando
TI  - Can Repeated Cross-Sections Replicate Panels? Optimal Synthetic Cohorts for Complex Survey Econometrics
PY  - 2026
DO  - 10.5281/zenodo.24326322
UR  - https://doi.org/10.5281/zenodo.24326322
ER  - 
```

---

## 📄 License

This software is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
