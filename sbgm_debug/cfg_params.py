from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# === Set up dataclasses for PathsParams, DataParams, TrainParams, EDMParams, RainGateParams ===

@dataclass(frozen=True)
class PathsParams:
    checkpoint_dir: str
    stats_dir: str
    path_save_root: str
    samples_dir: str
    figures_dir: str
    metrics_dir: str
    diagnostics_dir: str

@dataclass(frozen=True)
class DataParams:
    hr_var: str
    hr_scaling_method: str
    hr_model: str
    full_domain_dims_hr: Optional[List[int]]
    crop_region_hr: Optional[List[int]]
    lr_vars: List[str]
    lr_scaling_methods: List[str]
    lr_model: str
    full_domain_dims_lr: Optional[List[int]]
    crop_region_lr: Optional[List[int]]
    buffer_frac_hr: float
    buffer_frac_lr: float
    scaling_split: str
    prcp_eps: float

@dataclass(frozen=True)
class TrainParams:
    epochs: int
    mixed_precision: bool
    eval_use_ema: bool
    train_postfix_every: int
    verbose: bool

@dataclass(frozen=True)
class EDMParams:
    enabled: bool
    predict_residual: bool
    baseline_space: str  # "hr" | "lr"    

@dataclass(frozen=True)
class RainGateParams:
    enabled: bool
    include_lsm: bool
    include_topo: bool
    include_lr_baseline: bool
    wet_threshold_mm: float
    wet_threshold_model_space: float
    reweight_enabled: bool
    warm_start_epochs: int
    ramp_epochs: int
    loss_weight_bce: float
    pos_weight: float
    learning_rate: float
    c_hidden: int
    weight_strategy: str
    weight_alpha: float
    prob_gamma: float
    clip_max: float
    detach_weights: bool
    binary_threshold: float = 0.5  # Default threshold for binary classification

@dataclass(frozen=True)
class GuidanceParams:
    enabled: bool
    prob_drop_lr: float         # prob of dropping LR conditions
    drop_prob_geo: float        # prob of dropping static geo
    guidance_scale: float
    guidance_scale_max: float
    null_label_id: int        # for categorical cond (e.g., LSM)
    null_scalar_value: float  # for scalar cond (e.g
    sigma_weighted: bool
    drop_lr_ups_in_uncond: bool
    predict_residual: bool = False  # whether the model predicts residuals (for LR upsampling)

    @staticmethod
    def from_cfg(d: dict) -> "GuidanceParams":
        d = d or {}
        return GuidanceParams(
            enabled               = bool(d.get("enabled", False)),
            prob_drop_lr          = float(d.get("prob_drop_lr", d.get("drop_prob", 0.1))),  # keep compat with old 'drop_prob'
            drop_prob_geo         = float(d.get("drop_prob_geo", d.get("drop_prob", 0.1))),
            guidance_scale        = float(d.get("guidance_scale", 0.0)),
            guidance_scale_max    = float(d.get("guidance_scale_max", 1.0)),
            null_label_id         = int(d.get("null_label_id", 0)),
            null_scalar_value     = float(d.get("null_scalar_value", 0.0)),
            sigma_weighted        = bool(d.get("sigma_weighted", True)),
            drop_lr_ups_in_uncond = bool(d.get("drop_lr_ups_in_uncond", True)),
            predict_residual      = bool(d.get("predict_residual", False)),
        )

# --- Leaf dataclasses for clarity ---
@dataclass(frozen=True)
class DiagnosticsParams:
    per_batch_stats: bool
    log_every: int
    viz_every_n_epochs: int
    warn_if_abs_gt: float          # model-space residual clamp warning
    warn_if_phys_gt: float         # physical-space extreme warning (mm/day)

    @staticmethod
    def from_cfg(d: Dict[str, Any]) -> "DiagnosticsParams":
        d = d or {}
        return DiagnosticsParams(
            per_batch_stats   = bool(d.get("per_batch_stats", False)),
            log_every         = int(d.get("log_every", 100)),
            viz_every_n_epochs= int(d.get("viz_every_n_epochs", 10)),
            warn_if_abs_gt    = float(d.get("warn_if_abs_gt", 15.0)),
            warn_if_phys_gt   = float(d.get("warn_if_phys_gt", 350.0)),
        )

@dataclass(frozen=True)
class EndOfEpochParams:
    eval_batches: int
    fss_km: List[float]
    fss_threshold_mm: float
    psd_band: List[float]
    wet_day_threshold: float
    grid_km_per_px: float
    psd_compare_to_hr: bool
    quantiles_compare_to_hr: bool

    @staticmethod
    def from_cfg(d: Dict[str, Any]) -> "EndOfEpochParams":
        d = d or {}
        return EndOfEpochParams(
            eval_batches        = int(d.get("eval_batches", 1)),
            fss_km              = list(d.get("fss_km", [5, 10, 20, 50])),
            fss_threshold_mm    = float(d.get("fss_threshold_mm", 1.0)),
            psd_band            = list(d.get("psd_band", [0.05, 0.40])),
            wet_day_threshold   = float(d.get("wet_day_threshold", 1.0)),
            grid_km_per_px      = float(d.get("grid_km_per_px", 2.5)),
            psd_compare_to_hr   = bool(d.get("psd_compare_to_hr", True)),
            quantiles_compare_to_hr = bool(d.get("quantiles_compare_to_hr", True)),
        )

@dataclass(frozen=True)
class VisualizationsParams:
    enabled: bool
    transform_back_bf_plot: bool
    save: bool
    show: bool
    plot_every_n_epochs: int
    plot_every_n_steps: int
    plots_n_steps: List[str]
    plots_n_epoch: List[str]
    plots_other: List[str]
    n_gen_samples: int
    n_plot_samples: int
    gen_and_plot_every_n_epochs: int

    @staticmethod
    def from_cfg(d: Dict[str, Any]) -> "VisualizationsParams":
        d = d or {}
        return VisualizationsParams(
            enabled                 = bool(d.get("enabled", True)),
            transform_back_bf_plot  = bool(d.get("transform_back_bf_plot", True)),
            save                    = bool(d.get("save", True)),
            show                    = bool(d.get("show", False)),
            plot_every_n_epochs     = int(d.get("plot_every_n_epochs", 5)),
            plot_every_n_steps      = int(d.get("plot_every_n_steps", 1000)),
            plots_n_steps           = list(d.get("plots_n_steps", ["loss", "hr_lr_corr", "edm_cosine"])),
            plots_n_epoch           = list(d.get("plots_n_epoch", ["loss", "initial", "samples", "fss", "psd_slope", "quantiles", "weight_map"])),
            plots_other             = list(d.get("plots_other", ["initial", "samples"])),
            n_gen_samples           = int(d.get("n_gen_samples", 16)),
            n_plot_samples          = int(d.get("n_plot_samples", 4)),
            gen_and_plot_every_n_epochs = int(d.get("gen_and_plot_every_n_epochs", 1)),
        )

@dataclass(frozen=True)
class ExtremePrcpParams:
    enabled: bool
    threshold_mm: float
    back_transform: bool
    check_in_validation: bool
    clamp_in_generation: bool
    clamp_max_mm: Optional[float]

    @staticmethod
    def from_cfg(d: Dict[str, Any]) -> "ExtremePrcpParams":
        d = d or {}
        return ExtremePrcpParams(
            enabled              = bool(d.get("enabled", True)),
            threshold_mm         = float(d.get("threshold_mm", 500.0)),
            back_transform       = bool(d.get("back_transform", True)),
            check_in_validation  = bool(d.get("check_in_validation", True)),
            clamp_in_generation  = bool(d.get("clamp_in_generation", True)),
            clamp_max_mm         = float(d["clamp_max_mm"]) if "clamp_max_mm" in d else None,
        )

@dataclass(frozen=True)
class MonitoringParams:
    enabled: bool
    land_only: bool
    edm_metrics_every: int
    diagnostics: DiagnosticsParams
    visualizations: VisualizationsParams
    end_of_epoch: EndOfEpochParams
    extreme_prcp: ExtremePrcpParams

    @staticmethod
    def from_cfg(d: Dict[str, Any]) -> "MonitoringParams":
        d = d or {}
        vis = d.get("visualizations", {})
        eoe = d.get("end_of_epoch", d.get("end_of_epoch_evals", {}))  # keep compat with older key
        ext = d.get("extreme_prcp", {})
        diag = d.get("diagnostics_stats", d.get("diagnostics", {}))   # compat

        return MonitoringParams(
            enabled           = bool(d.get("enabled", True)),
            land_only         = bool(d.get("land_only", d.get("eval_land_only", True))),
            edm_metrics_every = int(d.get("edm_metrics_every", vis.get("plot_every_n_steps", 1000) // 10 if isinstance(vis.get("plot_every_n_steps", 1000), int) else 50)),
            diagnostics       = DiagnosticsParams.from_cfg(diag),
            visualizations    = VisualizationsParams.from_cfg(vis),
            end_of_epoch      = EndOfEpochParams.from_cfg(eoe),
            extreme_prcp      = ExtremePrcpParams.from_cfg(ext),
        )

    # Optional: validate plot names to catch typos early
    def validate(self, allowed_step: List[str], allowed_epoch: List[str]) -> "MonitoringParams":
        bad_step = [p for p in self.visualizations.plots_n_steps if p not in allowed_step]
        bad_epoch = [p for p in self.visualizations.plots_n_epoch if p not in allowed_epoch]
        if bad_step or bad_epoch:
            import warnings
            if bad_step:
                warnings.warn(f"[monitor] Unknown step plots: {bad_step}. Allowed: {allowed_step}")
            if bad_epoch:
                warnings.warn(f"[monitor] Unknown epoch plots: {bad_epoch}. Allowed: {allowed_epoch}")
        return self
@dataclass(frozen=True)
class SamplerParams:
    # High-level choice
    use_edm: bool                 # True => edm_sampler; False => one of {pc, EM, ODE}
    sampler_type: str             # "edm_sampler" | "pc_sampler" | "Euler_Maruyama_sampler" | "ode_sampler"

    # Common
    steps: int                    # total steps (EDM: sampling steps; others: total iterations)

    # --- EDM specifics (kept minimal & aligned with your config) ---
    P_mean: float
    P_std: float
    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float
    S_churn: float
    S_min: float
    S_max: float
    S_noise: float
    predict_residual: bool
    baseline_space: str           # "hr" | "lr"

    # --- Non-EDM samplers (optional knobs, safe defaults) ---
    snr: float                    # used by PC samplers (predictor-corrector SNR)
    n_steps_each: int             # corrector steps per level (PC)
    probability_flow: bool        # ODE flag

    @staticmethod
    def from_cfg(cfg: Dict[str, Any]) -> "SamplerParams":
        # Accept both old and new schema:
        edm_cfg = (cfg.get("edm") 
                   or cfg.get("model", {}).get("edm_params") 
                   or {})
        smp_cfg = cfg.get("sampler", {})  # non-EDM branch + legacy n_timesteps

        # Choose EDM if explicitly enabled OR sampler_type says "edm_sampler"
        sampler_type = str(smp_cfg.get("sampler_type", "edm_sampler"))
        use_edm = bool(edm_cfg.get("enabled", sampler_type == "edm_sampler"))

        # Unify steps
        steps = int(
            edm_cfg.get("sampling_steps",
                        smp_cfg.get("n_timesteps", 40))
        )

        # EDM defaults
        P_mean   = float(edm_cfg.get("P_mean", -1.5))
        P_std    = float(edm_cfg.get("P_std", 1.2))
        sigma_data = float(edm_cfg.get("sigma_data", 1.0))
        sigma_min  = float(edm_cfg.get("sigma_min", 2e-3))
        sigma_max  = float(edm_cfg.get("sigma_max", 80.0))
        rho        = float(edm_cfg.get("rho", 7.0))
        S_churn    = float(edm_cfg.get("S_churn", 0.0))
        S_min      = float(edm_cfg.get("S_min", 0.0))
        S_max      = float(edm_cfg.get("S_max", 0.0))
        S_noise    = float(edm_cfg.get("S_noise", 1.0))
        predict_residual = bool(edm_cfg.get("predict_residual", False))
        baseline_space   = str(edm_cfg.get("baseline_space", "hr"))

        # Non-EDM defaults
        snr            = float(smp_cfg.get("snr", 0.16))
        n_steps_each   = int(smp_cfg.get("n_steps_each", 1))
        probability_flow = bool(smp_cfg.get("probability_flow", False))

        return SamplerParams(
            use_edm=use_edm, sampler_type=sampler_type, steps=steps,
            P_mean=P_mean, P_std=P_std, sigma_data=sigma_data,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
            predict_residual=predict_residual, baseline_space=baseline_space,
            snr=snr, n_steps_each=n_steps_each, probability_flow=probability_flow
        )




