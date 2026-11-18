"""
dosc.dgp
========
Data Generating Processes (DGP) for simulating repeated cross-sections with
latent cohort structures, structural endogeneity, and longitudinal dynamics.

Author: Gustavo Fernando Huaman Fernandez
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def generate_synthetic_panel_dgp(
    N_t: int = 2000,
    T: int = 6,
    G_0: int = 8,
    delta: float = 2.0,
    beta_star: float = 0.5000,
    seed: int = 42
) -> Tuple[Dict[int, pd.DataFrame], List[str], List[str], str, str]:
    """Generate synthetic repeated cross-sections with known ground truth cohorts.

    Parameters
    ----------
    N_t : int, default=2000
        Number of cross-sectional observations per period.
    T : int, default=6
        Number of time waves/periods.
    G_0 : int, default=8
        True number of latent population cohorts.
    delta : float, default=2.0
        Separability factor between latent cohort centroids in invariant space Z.
    beta_star : float, default=0.5000
        True structural treatment effect parameter.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    time_series_data : dict of int -> pd.DataFrame
        Cross-sectional data per period.
    z_cols : list of str
        Names of invariant anchor variables.
    x_cols : list of str
        Names of structural explanatory variables.
    y_col : str
        Name of dependent outcome variable.
    w_col : str
        Name of sampling weight variable.
    """
    rng = np.random.default_rng(seed)
    d_z = 4

    # Latent cohort centroids with separation delta
    base = rng.standard_normal((G_0, d_z))
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    centroids = base * delta * 2.5
    alpha_star = np.linspace(-1.5, 1.5, G_0)
    gamma_star = np.linspace(-1.2, 1.2, G_0)
    psi_t_vec = np.linspace(-1.0, 1.0, T)

    time_data = {}
    z_cols = [f'Z_{k+1}' for k in range(d_z)]
    x_cols = ['X_1']
    y_col = 'Y'
    w_col = 'weight'

    for t in range(1, T + 1):
        true_g = rng.choice(G_0, size=N_t)

        noise_z = rng.standard_normal((N_t, d_z)) * 0.70
        Z = centroids[true_g] + noise_z

        alpha_i = alpha_star[true_g]
        gamma_i = gamma_star[true_g]
        psi_val = psi_t_vec[t - 1]
        nu_it = rng.standard_normal(N_t) * 1.2
        X_1 = 0.50 * alpha_i + gamma_i * psi_val + nu_it

        eps_it = rng.standard_normal(N_t) * 0.80
        Y = alpha_i + beta_star * X_1 + eps_it

        weights = np.exp(rng.normal(0.0, 0.35, size=N_t))

        df_dict = {
            'true_g': true_g + 1,
            'time': t,
            'Y': Y,
            'X_1': X_1,
            'weight': weights
        }
        for k, col in enumerate(z_cols):
            df_dict[col] = Z[:, k]

        time_data[t] = pd.DataFrame(df_dict)

    return time_data, z_cols, x_cols, y_col, w_col

