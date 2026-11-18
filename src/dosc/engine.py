"""
dosc.engine
===========
Core mathematical engine for Dynamic Optimal Synthetic Cohorts (DOSC).

Provides:
- PrototypeCatalog: Global invariant anchor profiles
- PseudoPanelResult: Optimization and pseudo-panel construction output
- DOSCEngine: Two-level optimization algorithm for synthetic cohort construction

Author: Gustavo Fernando Huaman Fernandez
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import DOSCConfig


@dataclass
class PrototypeCatalog:
    """Catalog of invariant global cohort prototypes Theta_G = {(mu_g, Sigma_g)}."""
    G: int
    centroids: np.ndarray             # Shape: (G, d)
    covariances: np.ndarray           # Shape: (G, d, d)
    precision_matrices: np.ndarray    # Shape: (G, d, d) - Regularized inverses
    global_weights: np.ndarray        # Shape: (G,) - Total population weight
    feature_names: List[str] = field(default_factory=list)
    scaler: Optional[StandardScaler] = None
    pca_model: Optional[PCA] = None


@dataclass
class PseudoPanelResult:
    """Structured result of the optimal pseudo-panel construction."""
    status: str                       # 'SUCCESS' or 'REJECT_NO_STABLE_COHORT_STRUCTURE'
    G_star: Optional[int]
    pseudo_panel_df: Optional[pd.DataFrame]
    prototypes: Optional[PrototypeCatalog]
    diagnostics_log: Dict[int, Dict[str, float]]
    selected_metrics: Optional[Dict[str, float]]
    rejection_reasons: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a readable executive summary of the optimization."""
        lines = ["=" * 60, "DOSC Optimization Summary", "=" * 60]
        lines.append(f"Status: {self.status}")
        if self.status == 'SUCCESS':
            lines.append(f"Optimal Cohorts (G*): {self.G_star}")
            if self.selected_metrics:
                m = self.selected_metrics
                lines.append(f"Distribution Loss (D_dist):    {m.get('d_dist', 0):.4f}")
                lines.append(f"Dependency Loss (D_dep):      {m.get('d_dep', 0):.4f}")
                lines.append(f"EIV Noise Ratio (D_eiv):      {m.get('d_eiv', 0):.4f}")
                lines.append(f"Temporal Stability (S_temp):  {m.get('s_temp', 0):.4f}")
                lines.append(f"Min Effective Sample (n_eff): {m.get('min_n_eff', 0):.1f}")
                lines.append(f"Total Multi-Loss:             {m.get('total_loss', 0):.4f}")
        else:
            lines.append("Rejection Reasons:")
            for r in self.rejection_reasons:
                lines.append(f" - {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


class DOSCEngine:
    """Two-level algorithmic engine for constructing Dynamic Optimal Synthetic Cohorts.

    Parameters
    ----------
    config : DOSCConfig, optional
        Hyperparameter configuration. If None, default DOSCConfig is used.
    """

    def __init__(self, config: Optional[DOSCConfig] = None):
        self.config = config or DOSCConfig()

    def _normalize_expansion_weights(
        self, 
        time_series_data: Dict[Any, pd.DataFrame], 
        weight_col: str
    ) -> Dict[Any, np.ndarray]:
        """Normalize sample expansion weights per period t to maintain equal scale:
        w_tilde_it = w_it * (N_t / sum_j w_jt)  =>  sum_i w_tilde_it = N_t.
        """
        normalized_weights = {}
        for t, df in time_series_data.items():
            N_t = len(df)
            w_raw = df[weight_col].to_numpy(dtype=float) if weight_col in df.columns else np.ones(N_t, dtype=float)
            sum_w_t = np.sum(w_raw)
            if sum_w_t <= 0:
                w_raw = np.ones(N_t, dtype=float)
                sum_w_t = float(N_t)
            scale = N_t / sum_w_t
            normalized_weights[t] = w_raw * scale
        return normalized_weights

    def fit_global_prototypes(
        self, 
        Z_pooled: np.ndarray, 
        weights_pooled: np.ndarray, 
        G: int
    ) -> PrototypeCatalog:
        """Level 1: Anchor Discovery. Fits G global invariant latent profiles
        over the pooled invariant anchor space Z_pooled.
        """
        d_orig = Z_pooled.shape[1]
        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z_pooled)
        pca_model = None
        Z_work = Z_scaled

        if (self.config.use_dim_reduction and d_orig > self.config.target_dim) or d_orig >= 20:
            corr_mat = np.corrcoef(Z_scaled, rowvar=False)
            eff_rank = int(np.sum(np.linalg.eigvalsh(corr_mat) > 0.25))
            if self.config.use_dim_reduction or eff_rank < d_orig // 2:
                n_comp = min(max(eff_rank, 4), self.config.target_dim, d_orig)
                pca_model = PCA(n_components=n_comp, random_state=42)
                Z_work = pca_model.fit_transform(Z_scaled)
        d = Z_work.shape[1]

        kmeans = KMeans(n_clusters=G, random_state=42, n_init=5, max_iter=200)
        labels = kmeans.fit_predict(Z_work, sample_weight=weights_pooled)
        centroids = kmeans.cluster_centers_

        covariances = np.zeros((G, d, d), dtype=float)
        precision_matrices = np.zeros((G, d, d), dtype=float)
        global_weights = np.zeros(G, dtype=float)
        eps_matrix = np.eye(d) * max(self.config.cov_epsilon, 1e-2)

        for g in range(G):
            mask = (labels == g)
            w_g = weights_pooled[mask]
            global_weights[g] = np.sum(w_g)

            if np.sum(mask) > 1 and np.sum(w_g) > 0:
                Z_g = Z_work[mask] - centroids[g]
                cov_g = np.cov(Z_g, rowvar=False, aweights=w_g)
                if cov_g.ndim == 0:
                    cov_g = np.array([[cov_g]])
                np.fill_diagonal(cov_g, np.maximum(np.diag(cov_g), 1e-2))
                cov_g = cov_g + eps_matrix
            else:
                cov_g = np.eye(d)

            try:
                prec_g = np.linalg.inv(cov_g)
            except np.linalg.LinAlgError:
                cov_g = cov_g + np.eye(d) * 1e-2
                prec_g = np.linalg.inv(cov_g)

            covariances[g] = cov_g
            precision_matrices[g] = prec_g

        return PrototypeCatalog(
            G=G,
            centroids=centroids,
            covariances=covariances,
            precision_matrices=precision_matrices,
            global_weights=global_weights,
            scaler=scaler,
            pca_model=pca_model
        )

    def compute_memberships(
        self, 
        Z_t: np.ndarray, 
        prototypes: PrototypeCatalog
    ) -> np.ndarray:
        """Compute fuzzy membership matrix p_ig for each unit in cross-section t.
        Soft assignment: Softmax over regularized Mahalanobis quadratic distances.
        Hard assignment: argmin of Mahalanobis distance.
        """
        if prototypes.scaler is not None:
            Z_work = prototypes.scaler.transform(Z_t)
            if prototypes.pca_model is not None:
                Z_work = prototypes.pca_model.transform(Z_work)
        else:
            Z_work = Z_t

        N = Z_work.shape[0]
        G = prototypes.G
        dist_sq = np.zeros((N, G), dtype=float)

        for g in range(G):
            diff = Z_work - prototypes.centroids[g]
            prec = prototypes.precision_matrices[g]
            dist_sq[:, g] = np.sum((diff @ prec) * diff, axis=1)

        if not self.config.soft_assignment:
            hard_clusters = np.argmin(dist_sq, axis=1)
            p_ig = np.zeros((N, G), dtype=float)
            p_ig[np.arange(N), hard_clusters] = 1.0
            return p_ig

        gamma = max(self.config.gamma_temperature, 1e-4)
        scaled_neg_dist = -0.5 * dist_sq / gamma
        max_val = np.max(scaled_neg_dist, axis=1, keepdims=True)
        exp_vals = np.exp(scaled_neg_dist - max_val)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        p_ig = exp_vals / np.maximum(sum_exp, 1e-12)
        return p_ig

    def project_and_aggregate_cross_section(
        self, 
        df_t: pd.DataFrame, 
        t_val: Any,
        z_cols: List[str], 
        x_cols: List[str], 
        y_col: str, 
        weight_col: str, 
        prototypes: PrototypeCatalog
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Level 2: Temporal Projection & Horvitz-Thompson Weighted Aggregation.
        Projects cross-section t onto the prototype catalog and generates synthetic cohort cells.
        """
        Z_t = df_t[z_cols].to_numpy(dtype=float)
        X_t = df_t[x_cols].to_numpy(dtype=float)
        Y_t = df_t[y_col].to_numpy(dtype=float)
        w_t = df_t[weight_col].to_numpy(dtype=float) if weight_col in df_t.columns else np.ones(len(df_t))

        p_ig = self.compute_memberships(Z_t, prototypes)
        G = prototypes.G

        rows = []
        eff_sizes = []

        for g in range(G):
            p_g = p_ig[:, g]
            eff_weights = w_t * p_g
            sum_eff_w = np.sum(eff_weights)
            sum_eff_w_sq = np.sum(eff_weights ** 2)

            if sum_eff_w > 1e-8:
                y_bar = np.sum(eff_weights * Y_t) / sum_eff_w
                x_bar = np.sum(eff_weights[:, None] * X_t, axis=0) / sum_eff_w
                z_bar = np.sum(eff_weights[:, None] * Z_t, axis=0) / sum_eff_w
                n_eff = (sum_eff_w ** 2) / max(sum_eff_w_sq, 1e-10)
                bessel = n_eff / max(n_eff - 1.0, 1.0) if n_eff > 1.0 else 1.0
                var_y_intra = bessel * (np.sum(eff_weights * ((Y_t - y_bar) ** 2)) / sum_eff_w)
            else:
                y_bar = 0.0
                x_bar = np.zeros(len(x_cols))
                z_bar = prototypes.centroids[g][:len(z_cols)] if len(prototypes.centroids[g]) >= len(z_cols) else np.zeros(len(z_cols))
                n_eff = 0.0
                bessel = 1.0
                var_y_intra = 0.0

            eff_sizes.append(n_eff)

            row_data = {
                'cohort_g': g + 1,
                'time_t': t_val,
                'Y_bar': y_bar,
                'W_pop': sum_eff_w,
                'n_eff': n_eff,
                'var_Y_intra': var_y_intra
            }
            for idx, col in enumerate(x_cols):
                x_key = col if col.startswith('X_') else f'X_{col}'
                var_x_key = f'var_{col}_intra' if col.startswith('X_') else f'var_X_{col}_intra'
                row_data[x_key] = x_bar[idx]
                if sum_eff_w > 1e-8:
                    var_x_j = bessel * (np.sum(eff_weights * ((X_t[:, idx] - x_bar[idx]) ** 2)) / sum_eff_w)
                else:
                    var_x_j = 0.0
                row_data[var_x_key] = var_x_j

            for idx, col in enumerate(z_cols):
                z_key = col if col.startswith('Z_') else f'Z_{col}'
                row_data[z_key] = z_bar[idx]

            rows.append(row_data)

        p_safe = np.clip(p_ig, 1e-12, 1.0)
        entropy_per_obs = -np.sum(p_safe * np.log(p_safe), axis=1) / max(np.log(G), 1e-4)
        df_pseudo_t = pd.DataFrame(rows)
        return df_pseudo_t, {'min_n_eff': min(eff_sizes), 'mean_entropy': float(np.mean(entropy_per_obs))}

    def evaluate_multi_loss(
        self, 
        time_series_data: Dict[Any, pd.DataFrame], 
        pseudo_panel_df: pd.DataFrame, 
        z_cols: List[str], 
        x_cols: List[str], 
        y_col: str, 
        weight_col: str, 
        prototypes: PrototypeCatalog
    ) -> Dict[str, float]:
        """Evaluate the multi-criteria loss metrics:
        1. D_dist: Normalized Wasserstein-1 marginal divergence
        2. D_dep: Frobenius cross-sectional covariance dependency loss
        3. D_eiv: Unsupervised errors-in-variables noise-to-signal attenuation ratio
        4. S_temp: Longitudinal silhouette temporal stability index
        """
        T = len(time_series_data)
        time_keys = sorted(list(time_series_data.keys()))

        # 1. D_dist
        w1_losses = []
        for t in time_keys:
            df_micro_t = time_series_data[t]
            df_pseudo_t = pseudo_panel_df[pseudo_panel_df['time_t'] == t]
            for col in x_cols:
                x_k = col if col.startswith('X_') else f'X_{col}'
                u_micro = df_micro_t[col].to_numpy(dtype=float)
                u_pseudo = df_pseudo_t[x_k].to_numpy(dtype=float)
                w_micro = df_micro_t[weight_col].to_numpy(dtype=float) if weight_col in df_micro_t.columns else None
                w_pseudo = df_pseudo_t['W_pop'].to_numpy(dtype=float)
                std_val = np.std(u_micro)
                if std_val > 1e-6:
                    w1 = wasserstein_distance(u_micro, u_pseudo, u_weights=w_micro, v_weights=w_pseudo) / std_val
                    w1_losses.append(w1)
        d_dist = float(np.mean(w1_losses)) if len(w1_losses) > 0 else 0.0

        # 2. D_dep
        frob_losses = []
        for t in time_keys:
            df_micro_t = time_series_data[t]
            df_pseudo_t = pseudo_panel_df[pseudo_panel_df['time_t'] == t]
            w_micro = df_micro_t[weight_col].to_numpy(dtype=float) if weight_col in df_micro_t.columns else None
            w_pseudo = df_pseudo_t['W_pop'].to_numpy(dtype=float)
            if len(x_cols) >= 2:
                cov_micro = np.cov(df_micro_t[x_cols].to_numpy(dtype=float), rowvar=False, aweights=w_micro)
                pseudo_x_cols = [c if c.startswith('X_') else f'X_{c}' for c in x_cols]
                cov_pseudo = np.cov(df_pseudo_t[pseudo_x_cols].to_numpy(dtype=float), rowvar=False, aweights=w_pseudo)
                norm_micro = np.linalg.norm(cov_micro, 'fro')
                if norm_micro > 1e-6:
                    frob_diff = np.linalg.norm(cov_micro - cov_pseudo, 'fro') / norm_micro
                    frob_losses.append(frob_diff)
        d_dep = float(np.mean(frob_losses)) if len(frob_losses) > 0 else 0.0

        # 3. D_eiv: Within-cell noise vs within-two-way signal
        p_df = pseudo_panel_df.copy()
        w_cell = p_df['W_pop'].to_numpy(dtype=float)
        sum_w = np.sum(w_cell)
        w_norm = w_cell / max(sum_w, 1e-8)
        noise_terms = []
        signal_terms = []

        time_dummies = pd.get_dummies(p_df['time_t'], prefix='t', drop_first=True, dtype=float)
        has_time = (time_dummies.shape[1] > 0)
        cohort_ids = p_df['cohort_g'].to_numpy()
        cohort_unique = np.unique(cohort_ids)

        if has_time:
            D_raw = time_dummies.to_numpy(dtype=float)
            D_dm = np.zeros_like(D_raw)
            for g in cohort_unique:
                mask = (cohort_ids == g)
                w_g = w_cell[mask]
                sw_g = max(np.sum(w_g), 1e-8)
                D_dm[mask] = D_raw[mask] - (np.sum(w_g[:, None] * D_raw[mask], axis=0) / sw_g)
            D_w = D_dm * np.sqrt(w_cell)[:, None]

        for col in x_cols:
            x_key = col if col.startswith('X_') else (f'X_{col}' if f'X_{col}' in p_df.columns else col)
            var_key = f'var_{col}_intra' if col.startswith('X_') else (f'var_X_{col}_intra' if f'var_X_{col}_intra' in p_df.columns else f'var_{col}_intra')
            if x_key in p_df.columns:
                x_raw = p_df[x_key].to_numpy(dtype=float)
                x_dm = np.zeros_like(x_raw)
                for g in cohort_unique:
                    mask = (cohort_ids == g)
                    w_g = w_cell[mask]
                    sw_g = max(np.sum(w_g), 1e-8)
                    x_dm[mask] = x_raw[mask] - (np.sum(w_g * x_raw[mask]) / sw_g)

                if has_time:
                    x_w = x_dm * np.sqrt(w_cell)
                    gamma = np.linalg.lstsq(D_w, x_w, rcond=None)[0]
                    x_tw_dm = x_dm - (D_dm @ gamma)
                else:
                    x_tw_dm = x_dm

                signal_terms.append(np.sum(w_norm * (x_tw_dm ** 2)))
                if var_key in p_df.columns:
                    var_arr = p_df[var_key].to_numpy(dtype=float)
                    neff_arr = np.maximum(p_df['n_eff'].to_numpy(dtype=float), 1.0)
                    noise_terms.append(np.sum(w_norm * (var_arr / neff_arr)))
                else:
                    noise_terms.append(0.0)

        eiv_ratios = [n / max(s, 1e-8) for n, s in zip(noise_terms, signal_terms)]
        d_eiv = float(np.mean(eiv_ratios)) if len(eiv_ratios) > 0 else 0.0

        # 4. S_temp
        if T >= 2:
            z_feat_cols = [col if col.startswith('Z_') else f'Z_{col}' for col in z_cols]
            drift_list = []
            var_list = []
            pooled_z = pseudo_panel_df[z_feat_cols].to_numpy(dtype=float)
            z_center = np.mean(pooled_z, axis=0)

            for idx in range(T - 1):
                t1, t2 = time_keys[idx], time_keys[idx + 1]
                z_t1 = pseudo_panel_df[pseudo_panel_df['time_t'] == t1].sort_values('cohort_g')[z_feat_cols].to_numpy(dtype=float)
                z_t2 = pseudo_panel_df[pseudo_panel_df['time_t'] == t2].sort_values('cohort_g')[z_feat_cols].to_numpy(dtype=float)
                drift = np.sum((z_t1 - z_t2) ** 2)
                variance = np.sum((z_t1 - z_center) ** 2)
                drift_list.append(drift)
                var_list.append(max(variance, 1e-6))

            relative_drifts = [d / v for d, v in zip(drift_list, var_list)]
            s_temp = float(np.clip(1.0 - np.mean(relative_drifts), 0.0, 1.0))
        else:
            s_temp = 1.0

        min_n_eff = float(pseudo_panel_df['n_eff'].min())

        return {
            'd_dist': d_dist,
            'd_dep': d_dep,
            'd_eiv': d_eiv,
            'd_inf': d_eiv,
            's_temp': s_temp,
            'min_n_eff': min_n_eff
        }

    def build_pseudo_panel_for_G(
        self, 
        time_series_data: Dict[Any, pd.DataFrame], 
        z_cols: List[str], 
        x_cols: List[str], 
        y_col: str, 
        weight_col: str, 
        G: int
    ) -> Tuple[pd.DataFrame, PrototypeCatalog, Dict[str, float]]:
        """Construct synthetic pseudo-panel and compute multi-criteria metrics for a fixed G."""
        normalized_weights = self._normalize_expansion_weights(time_series_data, weight_col)

        norm_time_series_data = {}
        for t, df_orig in time_series_data.items():
            df_norm = df_orig.copy()
            if weight_col in df_norm.columns and t in normalized_weights:
                df_norm[weight_col] = normalized_weights[t]
            norm_time_series_data[t] = df_norm

        all_z = []
        all_w = []
        for t in sorted(list(time_series_data.keys())):
            df = norm_time_series_data[t]
            all_z.append(df[z_cols].to_numpy(dtype=float))
            all_w.append(normalized_weights[t])

        Z_pooled = np.vstack(all_z)
        w_pooled = np.concatenate(all_w)

        prototypes = self.fit_global_prototypes(Z_pooled, w_pooled, G)

        t_dfs = []
        for t in sorted(list(time_series_data.keys())):
            df_t = norm_time_series_data[t]
            df_pseudo_t, _ = self.project_and_aggregate_cross_section(
                df_t, t, z_cols, x_cols, y_col, weight_col, prototypes
            )
            t_dfs.append(df_pseudo_t)

        pseudo_panel_df = pd.concat(t_dfs, ignore_index=True)
        metrics = self.evaluate_multi_loss(
            norm_time_series_data, pseudo_panel_df, z_cols, x_cols, y_col, weight_col, prototypes
        )
        return pseudo_panel_df, prototypes, metrics

    def optimize_G(
        self, 
        time_series_data: Dict[Any, pd.DataFrame], 
        z_cols: List[str], 
        x_cols: List[str], 
        y_col: str, 
        weight_col: str = 'weight'
    ) -> PseudoPanelResult:
        """Find the optimal number of cohorts G* minimizing the primal multi-criteria loss
        under the multi-dimensional feasibility constraint.
        """
        cfg = self.config
        diagnostics_log: Dict[int, Dict[str, float]] = {}
        feasible_G_list: List[int] = []
        panels_by_G: Dict[int, Tuple[pd.DataFrame, PrototypeCatalog]] = {}

        candidate_G_range = list(range(cfg.G_min, cfg.G_max + 1, cfg.step_size))

        for G in candidate_G_range:
            panel_df, prototypes, metrics = self.build_pseudo_panel_for_G(
                time_series_data, z_cols, x_cols, y_col, weight_col, G
            )
            is_feasible = (
                metrics['d_dist'] <= cfg.tau_d and
                metrics['d_dep'] <= cfg.tau_c and
                metrics['d_eiv'] <= cfg.tau_eiv and
                metrics['s_temp'] >= cfg.tau_s and
                metrics['min_n_eff'] >= cfg.n_min
            )
            metrics['is_feasible'] = is_feasible
            pen_term = float(cfg.complexity_penalty * (G / max(cfg.G_min, 1)))
            metrics['total_loss'] = float(metrics['d_dist'] + metrics['d_dep'] + metrics['d_eiv'] + (1.0 - metrics['s_temp']) + pen_term)
            diagnostics_log[G] = metrics
            panels_by_G[G] = (panel_df, prototypes)

            if is_feasible:
                feasible_G_list.append(G)
                if cfg.early_stop_on_first_feasible:
                    break

        if len(feasible_G_list) > 0:
            G_star = min(feasible_G_list, key=lambda g: diagnostics_log[g]['total_loss'])
            best_panel, best_proto = panels_by_G[G_star]
            return PseudoPanelResult(
                status='SUCCESS',
                G_star=G_star,
                pseudo_panel_df=best_panel,
                prototypes=best_proto,
                diagnostics_log=diagnostics_log,
                selected_metrics=diagnostics_log[G_star]
            )
        else:
            reasons = []
            best_s_temp = max((m['s_temp'] for m in diagnostics_log.values()), default=0.0)
            if best_s_temp < cfg.tau_s:
                reasons.append(f"Severe Temporal Instability: max S_temp = {best_s_temp:.3f} < tau_s ({cfg.tau_s})")
            best_n_eff = max((m['min_n_eff'] for m in diagnostics_log.values()), default=0.0)
            if best_n_eff < cfg.n_min:
                reasons.append(f"Insufficient Effective Cell Sample: max min_n_eff = {best_n_eff:.1f} < n_min ({cfg.n_min})")
            min_d_dist = min((m['d_dist'] for m in diagnostics_log.values()), default=1.0)
            if min_d_dist > cfg.tau_d:
                reasons.append(f"Excessive Distributional Loss: min D_dist = {min_d_dist:.3f} > tau_d ({cfg.tau_d})")
            min_d_eiv = min((m['d_eiv'] for m in diagnostics_log.values()), default=1.0)
            if min_d_eiv > cfg.tau_eiv:
                reasons.append(f"Excessive EIV Noise Ratio: min D_eiv = {min_d_eiv:.3f} > tau_eiv ({cfg.tau_eiv})")

            return PseudoPanelResult(
                status='REJECT_NO_STABLE_COHORT_STRUCTURE',
                G_star=None,
                pseudo_panel_df=None,
                prototypes=None,
                diagnostics_log=diagnostics_log,
                selected_metrics=None,
                rejection_reasons=reasons
            )

    def fit_dataframe(
        self,
        df: pd.DataFrame,
        time_col: str,
        z_cols: List[str],
        x_cols: List[str],
        y_col: str,
        weight_col: Optional[str] = None
    ) -> PseudoPanelResult:
        """High-level convenience method to fit DOSC directly from a single pandas DataFrame."""
        w_col = weight_col or '_dosc_w_internal'
        df_work = df.copy()
        if weight_col is None or weight_col not in df_work.columns:
            df_work[w_col] = 1.0

        time_series_data = {t: sub_df for t, sub_df in df_work.groupby(time_col)}
        return self.optimize_G(
            time_series_data=time_series_data,
            z_cols=z_cols,
            x_cols=x_cols,
            y_col=y_col,
            weight_col=w_col
        )

