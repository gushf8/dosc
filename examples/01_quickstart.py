"""
Example 01: Quickstart with DOSC
================================
Demonstrates how to generate repeated cross-sections, discover optimal
synthetic cohorts G*, estimate causal parameters via Synthetic WLS,
and test for statistical equivalence.
"""

from dosc import DOSCEngine, DOSCConfig, fit_synthetic_wls, two_one_sided_tost, generate_synthetic_panel_dgp

def main():
    print("=" * 70)
    print("DOSC QUICKSTART DEMO")
    print("=" * 70)

    # 1. Generate synthetic repeated cross-sections with known ground truth
    print("\n[1] Simulating repeated cross-sections (T=6 waves, N_t=2000, true beta=0.50)...")
    time_series_data, z_cols, x_cols, y_col, w_col = generate_synthetic_panel_dgp(
        N_t=2000,
        T=6,
        G_0=8,
        delta=2.0,
        beta_star=0.5000,
        seed=42
    )

    # 2. Configure and run DOSC optimization
    print("\n[2] Initializing DOSCEngine and optimizing cohort dimension G*...")
    config = DOSCConfig(
        G_min=4,
        G_max=16,
        step_size=2,
        tau_d=0.55,
        tau_c=0.25,
        tau_eiv=0.35,
        tau_s=0.85,
        n_min=15.0,
        gamma_temperature=1.0
    )
    engine = DOSCEngine(config=config)
    res = engine.optimize_G(
        time_series_data=time_series_data,
        z_cols=z_cols,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=w_col
    )

    print(res.summary())
    assert res.status == 'SUCCESS', "Optimization failed to find a feasible cohort structure"

    # 3. Fit Two-Way Fixed Effects Synthetic WLS
    print("\n[3] Estimating Two-Way FE model with Synthetic WLS and cluster-robust SE...")
    model = fit_synthetic_wls(
        pseudo_df=res.pseudo_panel_df,
        x_cols=x_cols,
        y_col='Y_bar',
        cohort_col='cohort_g',
        time_col='time_t',
        eiv_correction=True
    )
    print(model.summary())

    # 4. Statistical Equivalence Test (TOST) against true beta = 0.5000
    print("\n[4] Running Schuirmann TOST for parameter recovery...")
    beta_hat = float(model.coefficients[0])
    se_hat = float(model.standard_errors[0])
    tost = two_one_sided_tost(
        beta_est=beta_hat,
        beta_ref=0.5000,
        se_est=se_hat,
        df=model.degrees_of_freedom,
        delta=0.08
    )
    print(tost.summary())

    print("\n[SUCCESS] DOSC Quickstart completed successfully!")

if __name__ == "__main__":
    main()

