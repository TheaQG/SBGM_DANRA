# import torch

# from sbgm.evaluate_sbgm.metrics_univariate import (
#     pit_values_from_ensemble,
#     rank_histogram,
#     rxk_series,
#     fit_gev_block_maxima_with_ci,
#     fit_pot_gpd_with_ci,
#     seasonal_block_index,
#     )
# from sbgm.monitoring import (
#     compute_fss_at_scales,
#     compute_psd_slope,
#     compute_p95_p99_and_wet_day,
#     )
# class EvaluationRunner:
#     def __init__(self, cfg, paths, mask_hr=None, basins=None):
#         self.cfg = cfg
#         self.paths = paths
#         self.mask = mask_hr  # [H,W] or None
#         self.basins = basins # [H,W] int IDs or None

#     def _iter_days(self):
#         # yields dict(date, ens[M,1,H,W], pmm[1,1,H,W], hr[1,1,H,W], lr_hr[1,1,H,W])
#         for file in sorted((self.paths.ensembles_dir).glob("*.npz")):
#             d = load_npz(file); yield d

#     # ---------- Probabilistic/performance metrics (ensemble) ----------
#     def eval_probabilistic(self):
#         rows_crps = []
#         rows_reliab = []
#         rows_spread = []
#         for day in self._iter_days():
#             obs = torch.from_numpy(day['hr']).squeeze(0)      # [1,H,W] -> [H,W]
#             ens = torch.from_numpy(day['ens']).squeeze(1)     # [M,H,W]
#             msk = self.mask                                    # [H,W] or None

#             # CRPS: pixel-level mean over mask
#             crps_val = crps_ensemble(obs, ens, mask=msk, reduction="mean")
#             rows_crps.append({"date": day['date'], "crps": float(crps_val)})

#             # Reliability (exceedance) per threshold
#             for thr in self.cfg['evaluation']['thresholds_mm']:
#                 rel = reliability_exceedance_lr_binned(
#                         obs=obs, ens=ens, threshold=thr,
#                         lr_covariate=self._choose_lr_covariate(day), # [H,W] or [1] scalar per day
#                         n_bins=self.cfg['evaluation']['reliability_bins'],
#                         mask=msk, return_brier=True)
#                 # store bin-wise stats (prob_pred, freq_obs, count, brier) with the date & thr
#                 rows_reliab.append(serialize_rel(day['date'], thr, rel))

#             # Spread–skill with PMM as point estimator
#             ss = spread_skill(obs=obs, ens=ens, point="pmm", mask=msk, n_bins=self.cfg['evaluation']['spread_skill_bins'])
#             rows_spread.append(serialize_spread(day['date'], ss))

#         save_table("crps_daily.csv", rows_crps)
#         save_table("reliability_bins.csv", rows_reliab)
#         save_table("spread_skill.csv", rows_spread)

#         # Plots:
#         plot_crps_by_season("crps_daily.csv", seasons=self.cfg['data']['seasons'])
#         for thr in self.cfg['evaluation']['thresholds_mm']:
#             plot_reliability_from_table("reliability_bins.csv", thr)
#         plot_spread_skill("spread_skill.csv")

#     # ---------- Capability metrics (point field: PMM across all days) ----------
#     def eval_capability(self):
#         # Collect per-day PMM & HR
#         PMM, HR, dates = [], [], []
#         for day in self._iter_days():
#             PMM.append(torch.from_numpy(day['pmm']).squeeze(0))  # [1,H,W] -> [H,W]
#             HR.append(torch.from_numpy(day['hr']).squeeze(0))
#             dates.append(day['date'])
#         pmm = torch.stack(PMM, 0).unsqueeze(1)  # [B,1,H,W]
#         hr  = torch.stack(HR, 0).unsqueeze(1)   # [B,1,H,W]
#         msk = self.mask.unsqueeze(0).unsqueeze(1) if self.mask is not None else None

#         # FSS for thresholds × scales
#         for thr in self.cfg['evaluation']['thresholds_mm']:
#             scores = compute_fss_at_scales(gen_bt=pmm, hr_bt=hr, mask=msk,
#                                            grid_km_per_px=self.cfg['evaluation']['grid_km_per_px'],
#                                            fss_km=self.cfg['evaluation']['fss_scales_km'],
#                                            thr_mm=thr)
#             save_row("fss_summary.csv", {"thr": thr, **scores})

#         # Random reference FSS (baseline)
#         rand_pmm = self._random_reference_series(pmm, kind=self.cfg['evaluation']['random_ref']['type'])
#         for thr in self.cfg['evaluation']['thresholds_mm']:
#             scores_ref = compute_fss_at_scales(rand_pmm, hr, mask=msk,
#                                                grid_km_per_px=self.cfg['evaluation']['grid_km_per_px'],
#                                                fss_km=self.cfg['evaluation']['fss_scales_km'],
#                                                thr_mm=thr)
#             save_row("fss_random_ref.csv", {"thr": thr, **scores_ref})

#         # PSD slope & full PSDs (aggregate by season and all)
#         psd_summ = compute_psd_slope(gen_bt=pmm, hr_bt=hr, mask=msk,
#                                      ignore_low_k_bins=self.cfg['evaluation']['psd_ignore_low_k_bins'])
#         save_json("psd_slope_summary.json", psd_summ)
#         plot_psd_curves(pmm, hr, mask=msk, seasons=self.cfg['data']['seasons'])

#         # P95/P99 & wet-day freq
#         tails = compute_p95_p99_and_wet_day(pmm, hr_bt=hr, mask=msk, wet_threshold_mm=self.cfg['evaluation']['wet_threshold_mm'])
#         save_json("tails_summary.json", tails)

#         # PIT & Rank histograms (on ensemble vs obs — optional here, but good for the paper)
#         # Load again day by day to avoid storing whole ensemble
#         pit_vals_all = []
#         rank_counts_accum = None
#         for day in self._iter_days():
#             obs = torch.from_numpy(day['hr']).squeeze(0)        # [H,W]
#             ens = torch.from_numpy(day['ens']).squeeze(1)       # [M,H,W]
#             msk = self.mask
#             pit_vals_all.append(pit_values_from_ensemble(obs, ens, mask=msk))
#             rc = rank_histogram(obs, ens, mask=msk)
#             rank_counts_accum = rc if rank_counts_accum is None else (rank_counts_accum + rc)
#         save_hist("pit_hist.png", torch.cat(pit_vals_all), bins=self.cfg['evaluation']['pit_bins'])
#         plot_rank_hist(rank_counts_accum, out="rank_hist.png")

#     # ---------- Extremes (optional here or in a separate script) ----------
#     def eval_extremes(self):
#         # Build a daily basin-mean series from HR and from PMM (or sampled member)
#         series_hr  = to_numpy_1d_series(stack_days('hr'), mask=self.mask, agg="mean")
#         series_pmm = to_numpy_1d_series(stack_days('pmm'), mask=self.mask, agg="mean")
#         dates_np   = np.array([np.datetime64(d) for d in gather_dates()])

#         blk = seasonal_block_index(dates_np)
#         # GEV on Rx1day and Rx5day, seasonal blocks (block_per_year=4)
#         rx1_hr, _ = rxk_series(series_hr, 1, block_index=blk)
#         rx1_pmm,_ = rxk_series(series_pmm, 1, block_index=blk)
#         save_json("gev_rx1_hr.json",  fit_gev_block_maxima_with_ci(rx1_hr,  block_per_year=4.0))
#         save_json("gev_rx1_pmm.json", fit_gev_block_maxima_with_ci(rx1_pmm, block_per_year=4.0))
#         rx5_hr,_  = rxk_series(series_hr, 5, block_index=blk)
#         rx5_pmm,_ = rxk_series(series_pmm, 5, block_index=blk)
#         save_json("gev_rx5_hr.json",  fit_gev_block_maxima_with_ci(rx5_hr,  block_per_year=4.0))
#         save_json("gev_rx5_pmm.json", fit_gev_block_maxima_with_ci(rx5_pmm, block_per_year=4.0))

#         # POT/GPD (choose u, e.g. global or seasonal wet-day P95)
#         wet = series_hr[series_hr >= self.cfg['evaluation']['wet_threshold_mm']]
#         u = np.percentile(wet, 95.0)
#         save_json("pot_hr.json",  fit_pot_gpd_with_ci(series_hr,  threshold=u))
#         save_json("pot_pmm.json", fit_pot_gpd_with_ci(series_pmm, threshold=u))

#     # ---------- helpers ----------
#     def _choose_lr_covariate(self, day):
#         # Return [H,W] or [B] scalar for reliability LR-binning.
#         # Simplest: basin-mean LR scalar:
#         lr = torch.from_numpy(day['lr']).squeeze()  # [H,W]
#         if self.mask is not None:
#             return float(lr[self.mask].mean().item())  # [B] scalar style
#         else:
#             return float(lr.mean().item())

#     def _random_reference_series(self, pmm, kind="phase_randomized"):
#         # pmm: [B,1,H,W]  -> return random field with similar spectrum or marginal, per day.
#         if kind == "spatial_shuffle":
#             return shuffle_pixels_independently(pmm)
#         elif kind == "iid_marginal":
#             return iid_resample_same_marginal(pmm)
#         elif kind == "phase_randomized":
#             return fourier_phase_randomize(pmm)
#         else:
#             raise ValueError(kind)