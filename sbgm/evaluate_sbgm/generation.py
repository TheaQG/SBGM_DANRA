# import torch

# class GenerationRunner:
#     def __init__(self, model, dataloader_test, cfg, device):
#         self.model = model
#         self.dataloader_test = dataloader_test
#         self.cfg = cfg
#         self.device = device
#         self.rng = torch.Generator(device=device).manual_seed(cfg['generation']['sampler']['seed'])

#     @torch.no_grad()
#     def generate_one_day(self, batch):
#         # batch contains: lr_cond [1,Cl,Hl,Wl], optional aux (lsm_hr, topo, date, hr_truth if available)
#         # Upsample lr to HR grid if needed -> lr_hr [1,1,H,W]
#         # Build EDM sampling args (sigma_min chosen w.r.t sigma* if you use it)
#         # Produce ensemble: ens [M,1,H,W]
#         M = self.cfg['generation']['ensemble_size']
#         fields = []
#         for m in range(M):
#             x = edm_sample(self.model, lr_cond=batch.lr_hr, lsm=batch.lsm_hr, topo=batch.topo, rng=self.rng, cfg=self.cfg)
#             fields.append(x)  # x shape [1,1,H,W], back-transformed to mm/day
#         ens = torch.cat(fields, dim=0)              # [M,1,H,W]
#         pmm = pmm_from_ensemble(ens.unsqueeze(0)).squeeze(0)  # [1,1,H,W]
#         return ens, pmm 


#     def run(self, out_dir):
#         for i, batch in enumerate(self.loader):
#             ens, pmm = self.generate_one_day(batch)
#             date = batch.date_str  # "YYYY-MM-DD"
#             # Optionally also store HR truth for eval: y_true = batch.hr
#             save_npz(out_dir / "ensembles" / f"{date}.npz",
#                      ens=to_cpu(ens).numpy(),
#                      pmm=to_cpu(pmm).numpy(),
#                      lr=to_cpu(batch.lr_hr).numpy(),
#                      hr=to_cpu(batch.hr).numpy() if hasattr(batch, 'hr') else None)
#             if i % self.cfg['output']['log_every'] == 0:
#                 log.info(f"Generated {i}/{len(self.loader)}")
        