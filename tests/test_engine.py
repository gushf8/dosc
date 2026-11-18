"""
Unit tests for DOSC engine and estimators.
"""

import pytest
import numpy as np
from dosc import (
    DOSCEngine, 
    DOSCConfig, 
    fit_synthetic_wls, 
    two_one_sided_tost, 
    generate_synthetic_panel_dgp
)

def test_dgp_generation():
    time_series_data, z_cols, x_cols, y_col, w_col = generate_synthetic_panel_dgp(
        N_t=500, T=3, G_0=4, delta=2.0, beta_star=0.50, seed=42
    )
    assert len(time_series_data) == 3
    assert len(z_cols) == 4
    assert len(x_cols) == 1
    for t, df in time_series_data.items():
        assert len(df) == 500
        assert y_col in df.columns
        assert w_col in df.columns

def test_dosc_optimization_and_wls():
    time_series_data, z_cols, x_cols, y_col, w_col = generate_synthetic_panel_dgp(
        N_t=1000, T=4, G_0=6, delta=2.5, beta_star=0.50, seed=123
    )
    cfg = DOSCConfig(
        G_min=4,
        G_max=10,
        step_size=2,
        tau_d=0.60,
        tau_c=0.30,
        tau_eiv=0.40,
        tau_s=0.80,
        n_min=10.0
    )
    engine = DOSCEngine(config=cfg)
    res = engine.optimize_G(
        time_series_data=time_series_data,
        z_cols=z_cols,
        x_cols=x_cols,
        y_col=y_col,
        weight_col=w_col
    )
    assert res.status == 'SUCCESS'
    assert res.G_star in [4, 6, 8, 10]
    assert res.pseudo_panel_df is not None

    model = fit_synthetic_wls(
        pseudo_df=res.pseudo_panel_df,
        x_cols=x_cols,
        y_col='Y_bar'
    )
    assert len(model.coefficients) == 1
    assert abs(model.coefficients[0] - 0.50) < 0.15

def test_tost_equivalence():
    tost = two_one_sided_tost(
        beta_est=0.502,
        beta_ref=0.500,
        se_est=0.03,
        df=25,
        delta=0.08
    )
    assert tost.is_equivalent
    assert tost.p_tost < 0.05

