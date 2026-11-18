"""
Example 02: Benchmark DOSC vs Traditional Deaton (1985)
======================================================
Compares the performance of data-driven DOSC against fixed-interval
Deaton (1985) synthetic cohorts in recovering ground truth parameters.
"""

import numpy as np
import pandas as pd
from dosc import (
    DOSCEngine, 
    DOSCConfig, 
    fit_synthetic_wls, 
    fit_deaton_cohorts, 
    generate_synthetic_panel_dgp
)

def main():
    print("=" * 70)
    print("DOSC VS DEATON (1985) BENCHMARK")
    print("=" * 70)

    # 1. Simulate data where Z_1 acts as a proxy for birth cohort/age
    time_series_data, z_cols, x_cols, y_col, w_col = generate_synthetic_panel_dgp(
        N_t=2500,
        T=8,
        G_0=8,
        delta=2.0,
        beta_star=0.5000,
        seed=123
    )

    # Discretize Z_1 into 8 fixed bins to emulate classical Deaton cohorts
    for t, df in time_series_data.items():
        df['deaton_cohort'] = pd.qcut(df['Z_1'], q=8, labels=False)

    print("\n[1] Fitting classical Deaton (1985) fixed-cell pseudopanel...")
    deaton_model = fit_deaton_cohorts(
        time_series_data=time_series_data,
        cohort_var='deaton_cohort',
        x_cols=x_cols,
        y_col=y_col,
        weight_col=w_col
    )
    b_deaton = float(deaton_model.coefficients[0])
    se_deaton = float(deaton_model.standard_errors[0])
    bias_deaton = abs(b_deaton - 0.5000)

    print(f"  Deaton (1985): beta_hat = {b_deaton:.4f}, SE = {se_deaton:.4f}, Bias = {bias_deaton:.4f}")

    print("\n[2] Fitting DOSC with endogenous dimension selection and soft assignment...")
    engine = DOSCEngine(DOSCConfig(G_min=4, G_max=16, step_size=2))
    res = engine.optimize_G(
        time_series_data=time_series_data,
        z_cols=z_cols,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=w_col
    )
    assert res.status == 'SUCCESS'
    dosc_model = fit_synthetic_wls(
        pseudo_df=res.pseudo_panel_df,
        x_cols=x_cols,
        y_col='Y_bar',
        eiv_correction=True
    )
    b_dosc = float(dosc_model.coefficients[0])
    se_dosc = float(dosc_model.standard_errors[0])
    bias_dosc = abs(b_dosc - 0.5000)

    print(f"  DOSC (G*={res.G_star}): beta_hat = {b_dosc:.4f}, SE = {se_dosc:.4f}, Bias = {bias_dosc:.4f}")

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS:")
    print(f"  True Parameter:  0.5000")
    print(f"  Deaton (1985):   {b_deaton:.4f} (Bias: {bias_deaton:.4f}, SE: {se_deaton:.4f})")
    print(f"  DOSC (Proposed): {b_dosc:.4f} (Bias: {bias_dosc:.4f}, SE: {se_dosc:.4f})")
    if bias_dosc < bias_deaton:
        print(f"  -> DOSC achieves a {((bias_deaton - bias_dosc) / bias_deaton) * 100:.1f}% reduction in bias!")
    print("=" * 70)

if __name__ == "__main__":
    main()

