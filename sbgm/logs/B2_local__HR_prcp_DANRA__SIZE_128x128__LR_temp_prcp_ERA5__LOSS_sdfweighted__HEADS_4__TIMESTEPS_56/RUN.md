# Run manifest: B2_local__67938858__20260129T171457Z

```
{
  "timestamp": "2026-01-29T17:14:57.218557Z",
  "run_name": "B2_local__67938858__20260129T171457Z",
  "model_name": "B2_local__HR_prcp_DANRA__SIZE_128x128__LR_temp_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56",
  "cfg_hash": "67938858",
  "paths": {
    "data_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/../../Data/data_DiffMod_small",
    "checkpoint_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/models_and_samples/trained_models",
    "checkpoint_name": "sbgm_cfgTest.pth.tar",
    "sample_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/models_and_samples/generated_samples",
    "evaluation_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/evaluate_sbgm/results",
    "log_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/sbgm/logs",
    "specific_fig_name": "test__plot_fct",
    "path_save": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/models_and_samples/generated_samples",
    "lsm_path": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/../../Data/data_DiffMod_small/data_lsm/truth_fullDomain/lsm_full.npz",
    "topo_path": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/../../Data/data_DiffMod_small/data_topo/truth_fullDomain/topo_full.npz",
    "slope_path": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/../../Data/data_DiffMod_small/data_slope/truth_fullDomain/slope_full.npz",
    "stats_load_dir": "/Users/au728490/OneDrive - Aarhus universitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_SD/data_analysis_pipeline/saved/statistics_run/stats"
  },
  "experiment": {
    "name": "B2_local",
    "config_name": "B2_local",
    "date": "20260129"
  },
  "training": {
    "seed": 504,
    "device": "cpu",
    "use_mixed_precision": false,
    "use_grad_clip": false,
    "grad_clip_norm": 0.0,
    "verbose": true,
    "batch_size": 4,
    "learning_rate": 0.0002,
    "min_lr": 5e-07,
    "lr_scheduler": "ReduceLROnPlateau",
    "lr_scheduler_params": {
      "factor": 0.5,
      "patience": 15,
      "threshold": 0.01,
      "min_lr": 1e-06
    },
    "weight_init": true,
    "custom_weight_initializer": null,
    "with_ema": false,
    "eval_use_ema": false,
    "load_ema": false,
    "ema_decay": 0.999,
    "weight_decay": 1e-06,
    "ema_warmup_steps": 2000,
    "epochs": 2,
    "loss_type": "sdfweighted",
    "sdf_weighted_loss": true,
    "optimizer": "adamw",
    "load_checkpoint": false,
    "early_stopping": true,
    "early_stopping_params": {
      "patience": 75,
      "min_delta": 0.0001
    },
    "train_use_amp": false,
    "train_postfix_every": 10,
    "detect_anomaly": false,
    "debug_device_asserts": false,
    "debug_pre_sigma_div": false
  },
  "edm": {
    "enabled": true,
    "P_mean": -1.5,
    "P_std": 1.2,
    "sigma_data": 1.0,
    "sigma_min": 0.002,
    "sigma_star": 1.0,
    "sigma_max": 80,
    "rho": 7.0,
    "S_churn": 2.0,
    "S_min": 40.0,
    "S_max": 80.0,
    "S_noise": 1.0,
    "sampling_steps": 56,
    "predict_residual": false,
    "baseline_space": "hr",
    "drop_lr_ups_in_uncond": true
  },
  "highres": {
    "model": "DANRA",
    "variable": "prcp",
    "data_size": [
      128,
      128
    ],
    "scaling_method": "log_zscore",
    "full_domain_dims": [
      589,
      789
    ],
    "buffer_frac": 0.0,
    "cutout_domains": [
      170,
      350,
      340,
      520
    ],
    "cutout_name": "danra",
    "stationary_cutout": {
      "enabled": false,
      "bounds": [
        200,
        328,
        380,
        508
      ]
    }
  },
  "lowres": {
    "model": "ERA5",
    "full_domain_dims": [
      589,
      789
    ],
    "condition_variables": [
      "temp",
      "prcp"
    ],
    "scaling_methods": [
      "zscore",
      "log_zscore"
    ],
    "dual_lr": false,
    "lr_main_var_scale": "LR",
    "lr_main_var_scale_method": "log_zscore",
    "buffer_frac": 0.0,
    "data_size": [
      128,
      128
    ],
    "resize_factor": 1,
    "cutout_domains": [
      170,
      350,
      340,
      520
    ],
    "cutout_name": "danra",
    "stationary_cutout": {
      "enabled": false,
      "bounds": [
        200,
        328,
        380,
        508
      ]
    }
  }
}
```
