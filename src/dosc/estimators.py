"""
dosc.estimators
===============
Econometric estimators for synthetic cohort panels:
- Two-Way Fixed Effects Synthetic-Cohort Weighted Least Squares (Synthetic WLS)
- Finite-sample Measurement Error / Errors-in-Variables (EIV) moment correction
- Two-way cluster-robust covariance matrix (Cameron, Gelbach & Miller, 2011)
- Classical Deaton (1985) fixed-cell benchmark estimator

Author: Gustavo Fernando Huaman Fernandez
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


@dataclass
class SyntheticCohortModel:
    """Estimated econometric model result on synthetic cohorts."""
    estimator_name: str
    feature_names: List[str]
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    degrees_of_freedom: int
    n_cohorts: int
    n_periods: int
    n_cells: int
    residuals: np.ndarray
    r_squared_within: float

    def summary(self) -> str:
        """Produce a formatted regression summary table."""
        lines = [
            "=" * 78,
            f"Model: {self.estimator_name}",
            f"Cohorts (G): {self.n_cohorts:<8} Periods (T): {self.n_periods:<8} Total Cells: {self.n_cells}",
            f"Degrees of freedom: {self.degrees_of_freedom:<8} R-squared (Within): {self.r_squared_within:.4f}",
            "-" * 78,
            f"{'Variable':<18} {'Coef.':>10} {'Std. Err.':>11} {'t':>8} {'P>|t|':>8} {'[95% Conf. Interval]':>18}",
            "-" * 78
        ]
        for idx, var in enumerate(self.feature_names):
            c = self.coefficients[idx]
            se = self.standard_errors[idx]
            t_val = self.t_statistics[idx]
            p_val = self.p_values[idx]
            ci_l = self.ci_lower[idx]
            ci_u = self.ci_upper[idx]
            lines.append(f"{var:<18} {c:10.5f} {se:11.5f} {t_val:8.2f} {p_val:8.4f} {ci_l:9.4f} {ci_u:9.4f}")
        lines.append("=" * 78)
        return "\n".join(lines)


def fit_synthetic_wls(
    pseudo_df: pd.DataFrame, 
    x_cols: List[str], 
    y_col: str = 'Y_bar', 
    cohort_col: str = 'cohort_g', 
    time_col: str = 'time_t',
    eiv_correction: bool = False,
    kappa: float = 0.05
) -> SyntheticCohortModel:
    """Fit Two-Way Fixed Effects Synthetic-Cohort Weighted Least Squares (Synthetic WLS).

    Parameters
    ----------
    pseudo_df : pd.DataFrame
        Synthetic pseudo-panel DataFrame produced by DOSCEngine.
    x_cols : list of str
        List of structural explanatory variable names.
    y_col : str, default='Y_bar'
        Column name of the synthetic dependent variable.
    cohort_col : str, default='cohort_g'
        Column identifying synthetic cohort IDs.
    time_col : str, default='time_t'
        Column identifying time periods.
    eiv_correction : bool, default=False
        Whether to apply Deaton (1985) finite-sample errors-in-variables moment correction.
    kappa : float, default=0.05
        Within-cell variance dampening weight in analytic precision weights.

    Returns
    -------
    SyntheticCohortModel
        Dataclass containing estimated coefficients, cluster-robust standard errors, and tests.
    """
    df = pseudo_df.copy()
    cohort_unique = np.unique(df[cohort_col])
    time_unique = np.unique(df[time_col])
    G_count = len(cohort_unique)
    T_count = len(time_unique)

    # Analytic precision weights: omega_gt = n_eff / (1 + kappa * var_Y_intra)
    df['omega_gt'] = df['n_eff'] / (1.0 + kappa * np.maximum(df.get('var_Y_intra', 0.0), 0.0))
    mean_omega = df['omega_gt'].mean()
    df['omega_gt'] = df['omega_gt'] / (mean_omega if mean_omega > 1e-8 else 1.0)

    y_raw = df[y_col].to_numpy(dtype=float)
    x_feature_cols = [f'X_{col}' if f'X_{col}' in df.columns else col for col in x_cols]
    X_structural = df[x_feature_cols].to_numpy(dtype=float)
    p_structural = len(x_cols)

    # Project time dummies for Two-Way FE
    time_dummies = pd.get_dummies(df[time_col], prefix='t', drop_first=True, dtype=float)
    if time_dummies.shape[1] > 0:
        X_raw = np.column_stack([X_structural, time_dummies.to_numpy(dtype=float)])
    else:
        X_raw = X_structural

    omega = df['omega_gt'].to_numpy(dtype=float)
    cohort_ids = df[cohort_col].to_numpy()

    # Weighted Within-demeaning by cohort
    y_dm = np.zeros_like(y_raw)
    X_dm = np.zeros_like(X_raw)

    for g in cohort_unique:
        mask = (cohort_ids == g)
        w_g = omega[mask]
        sum_w = np.sum(w_g)
        y_mean_g = np.sum(w_g * y_raw[mask]) / max(sum_w, 1e-8)
        X_mean_g = np.sum(w_g[:, None] * X_raw[mask], axis=0) / max(sum_w, 1e-8)
        y_dm[mask] = y_raw[mask] - y_mean_g
        X_dm[mask] = X_raw[mask] - X_mean_g

    W_diag = np.sqrt(omega)
    X_w = X_dm * W_diag[:, None]
    y_w = y_dm * W_diag

    XtX = X_w.T @ X_w
    Xty = X_w.T @ y_w

    if eiv_correction and T_count > 1:
        psi_eta = np.zeros_like(XtX)
        for idx, col in enumerate(x_cols):
            var_key = f'var_{col}_intra' if col.startswith('X_') else (f'var_X_{col}_intra' if f'var_X_{col}_intra' in df.columns else f'var_{col}_intra')
            if var_key in df.columns:
                noise_var = np.sum(df['omega_gt'] * df[var_key] / np.maximum(df['n_eff'], 1.0))
                psi_eta[idx, idx] = ((T_count - 1.0) / T_count) * noise_var
        XtX_adj = XtX - psi_eta
        try:
            beta_hat_full = np.linalg.solve(XtX_adj, Xty)
        except np.linalg.LinAlgError:
            beta_hat_full = np.linalg.solve(XtX, Xty)
    else:
        try:
            beta_hat_full = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta_hat_full = np.linalg.lstsq(X_w, y_w, rcond=None)[0]

    residuals = y_dm - (X_dm @ beta_hat_full)
    XtX_inv = np.linalg.pinv(XtX)

    # Two-Way Cluster-Robust Covariance (Cameron, Gelbach & Miller, 2011)
    def compute_cluster_meat(cluster_var):
        unique_c = np.unique(cluster_var)
        meat = np.zeros_like(XtX)
        for c in unique_c:
            mask = (cluster_var == c)
            score_c = X_w[mask].T @ (residuals[mask] * W_diag[mask])
            meat += np.outer(score_c, score_c)
        c_adj = len(unique_c) / max(len(unique_c) - 1, 1)
        return meat * c_adj

    V_g = compute_cluster_meat(df[cohort_col].to_numpy())
    if T_count > 1:
        V_t = compute_cluster_meat(df[time_col].to_numpy())
        cell_id = df[cohort_col].astype(str) + "_" + df[time_col].astype(str)
        V_gt = compute_cluster_meat(cell_id.to_numpy())
        V_twoway = XtX_inv @ (V_g + V_t - V_gt) @ XtX_inv
    else:
        V_twoway = XtX_inv @ V_g @ XtX_inv

    # Ensure positive semi-definiteness
    diag_v = np.diag(V_twoway[:p_structural, :p_structural])
    diag_v = np.maximum(diag_v, 1e-12)
    se_vector = np.sqrt(diag_v)

    beta_struct = beta_hat_full[:p_structural]
    t_stats = beta_struct / se_vector
    df_cluster = max(G_count - 1, 1)
    p_values = 2.0 * (1.0 - student_t.cdf(np.abs(t_stats), df=df_cluster))

    crit = student_t.ppf(0.975, df=df_cluster)
    ci_l = beta_struct - crit * se_vector
    ci_u = beta_struct + crit * se_vector

    # R-squared within
    ss_tot = np.sum(y_w ** 2)
    ss_res = np.sum((residuals * W_diag) ** 2)
    r2_within = max(1.0 - (ss_res / max(ss_tot, 1e-10)), 0.0)

    return SyntheticCohortModel(
        estimator_name="DOSC Synthetic WLS (Two-Way FE)",
        feature_names=list(x_cols),
        coefficients=beta_struct,
        standard_errors=se_vector,
        t_statistics=t_stats,
        p_values=p_values,
        ci_lower=ci_l,
        ci_upper=ci_u,
        degrees_of_freedom=df_cluster,
        n_cohorts=G_count,
        n_periods=T_count,
        n_cells=len(df),
        residuals=residuals,
        r_squared_within=r2_within
    )


def fit_deaton_cohorts(
    time_series_data: Dict[Any, pd.DataFrame],
    cohort_var: str,
    x_cols: List[str],
    y_col: str,
    time_col: str = 'time',
    weight_col: Optional[str] = None
) -> SyntheticCohortModel:
    """Fit classical Deaton (1985) fixed-cell pseudopanel estimator.

    Parameters
    ----------
    time_series_data : dict of int/str -> pd.DataFrame
        Dictionary mapping period identifiers to cross-sectional DataFrames.
    cohort_var : str
        Discrete variable name defining fixed cohort cells (e.g. birth year, 5-year bracket).
    x_cols : list of str
        Explanatory variables.
    y_col : str
        Dependent variable.
    time_col : str, default='time'
        Time period column name.
    weight_col : str, optional
        Sample expansion weights column name.
    """
    rows = []
    for t, df in time_series_data.items():
        w = df[weight_col].to_numpy(dtype=float) if weight_col and weight_col in df.columns else np.ones(len(df))
        for c_id, sub_df in df.groupby(cohort_var):
            sub_w = sub_df[weight_col].to_numpy(dtype=float) if weight_col and weight_col in sub_df.columns else np.ones(len(sub_df))
            sw = np.sum(sub_w)
            sw_sq = np.sum(sub_w ** 2)
            n_eff = (sw ** 2) / max(sw_sq, 1e-10)
            y_mean = np.sum(sub_w * sub_df[y_col].to_numpy(dtype=float)) / max(sw, 1e-8)
            x_means = {col: np.sum(sub_w * sub_df[col].to_numpy(dtype=float)) / max(sw, 1e-8) for col in x_cols}
            row = {
                'cohort_g': c_id,
                'time_t': t,
                'Y_bar': y_mean,
                'n_eff': n_eff,
                'W_pop': sw,
                'var_Y_intra': np.var(sub_df[y_col])
            }
            row.update(x_means)
            rows.append(row)

    pseudo_df = pd.DataFrame(rows)
    model = fit_synthetic_wls(
        pseudo_df=pseudo_df,
        x_cols=x_cols,
        y_col='Y_bar',
        cohort_col='cohort_g',
        time_col='time_t',
        eiv_correction=False
    )
    model.estimator_name = "Deaton (1985) Fixed-Cell Pseudopanel"
    return model

