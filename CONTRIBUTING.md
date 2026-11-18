# Contributing to DOSC Research

Thank you for your interest in the **Dynamic Optimal Synthetic Cohorts (DOSC)** research framework and replication repository.

This repository serves as the official open-science replication package for the research paper:  
*“Dynamic Optimal Synthetic Cohorts (DOSC): Resolving Cardinality, Heterogeneity, and Measurement Bias in Repeated Cross-Sections”*.

---

## 🔬 Replication Protocol

All analytical routines, Monte Carlo benchmarks, and empirical validations are fully deterministic.

### Prerequisites
1. Python 3.9+ with scientific packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Full automated replication:
   ```bash
   python run_all_dosc_replication.py --seed 42
   ```

Outputs and verification metrics will populate:
- `results/*.json`: Quantitative benchmark logs and test statistics.
- `09_ENTREGABLES_FINALES/02_PAQUETE_SUBMISSION_ARXIV/figures/*.png`: Scientific charts at 300 DPI.

---

## 🛠️ Code and Methodology Contributions

We welcome feedback, issues, and pull requests from economists, econometricians, and data scientists:
1. **Reporting Bugs or Discrepancies:** Open an issue with full terminal trace, Python version, OS platform, and seed parameter.
2. **Extensions to Other Longitudinal Contexts:** If you wish to adapt DOSC to other surveys (e.g., CPS, ENIGH, CASEN, PNAD), consider sub-classing `DOSCEngine` in `scripts/01_dosc_engine.py`.
3. **Pull Requests:** Ensure all code adheres to PEP 8, maintains zero regressions across Monte Carlo ground truth benchmarks, and does not alter the canonical empirical results.

---

## 📜 Scientific Citation
If you use this codebase or algorithm, please cite the framework using the metadata provided in `CITATION.cff`.
