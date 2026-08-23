# Paper checkpoints

Source of truth for the trained checkpoints used by Tables 1 and 2 in
`main_arxiv.tex`. Table 2 reuses the 750M and 1.3B checkpoints below; the
throughput figure does not use a checkpoint.

Path and cluster abbreviations:

- `OUT=/checkpoints/ngocbh/hdn/ckpts/outputs`
- `MF=manifold://deep_retrieval/tree/datasets/ngocbh_hdn/ckpts`
- `T20-H200=shared-aws-usw1-1`; `MAST-H100=grandteton`; `MAST-B200=grandteton_b200`
- GPU allocations are written as nodes x GPUs per node.

| Paper block | Model | Run ID | Training cluster / job | Tracker | Final checkpoint |
|---|---|---|---|---|---|
| 220M / 20B | DeltaNet | `repro_ngocbh_deltanet_220M_20B_0718_0136` | T20-H200, 1x8, SLURM `1539011` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_deltanet_220M_20B_0718_0136) | `$OUT/tsz128x4k_20B_repro_ngocbh_deltanet_220M_20B_0718_0136/final-model-ckpt.pth` |
| 220M / 20B | Gated DeltaNet | `repro_ngocbh_gateddeltanet_220M_20B_0718_0131` | T20-H200, 1x8, SLURM `1538989` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gateddeltanet_220M_20B_0718_0131) | `$OUT/tsz128x4k_20B_repro_ngocbh_gateddeltanet_220M_20B_0718_0131/final-model-ckpt.pth` |
| 220M / 20B | GDN-2 | `repro_ngocbh_gdn2_220M_20B_0714_0317` | T20-H200, 1x8, SLURM `1521127` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_220M_20B_0714_0317) | `$OUT/tsz128x4k_20B_repro_ngocbh_gdn2_220M_20B_0714_0317/final-model-ckpt.pth` |
| 220M / 20B | KDA | `repro_ngocbh_kda_220M_20B_0718_0338` | T20-H200, 1x8, SLURM `1539391` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_220M_20B_0718_0338) | `$OUT/tsz128x4k_20B_repro_ngocbh_kda_220M_20B_0718_0338/final-model-ckpt.pth` |
| 220M / 20B | Isotropic KDN | `repro_ngocbh_iso_kla_220M_20B_0724_1833` | T20-H200, 1x8, SLURM `1564904`, `1565202` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_220M_20B_0724_1833) | `$OUT/tsz128x4k_20B_repro_ngocbh_iso_kla_220M_20B_0724_1833/final-model-ckpt.pth` |
| 220M / 20B | Diagonal KDN | `repro_ngocbh_diag_kla_220M_20B_job1565893` | T20-H200, 1x8, SLURM `1565893` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_220M_20B_job1565893) | `$OUT/tsz128x4k_20B_repro_ngocbh_diag_kla_220M_20B_job1565893/final-model-ckpt.pth` |
| 750M / 50B | GDN-2 | `repro_ngocbh_gdn2_750M_50B_job1576159` | T20-H200, 1x8, SLURM `1576159`, `1580790` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_750M_50B_job1576159) | `$OUT/tsz128x4k_50B_repro_ngocbh_gdn2_750M_50B_job1576159/final-model-ckpt.pth` |
| 750M / 50B | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1693970` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_siso_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1694025` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_mimo_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | KDA | `repro_ngocbh_kda_750M_50B_job1576160` | T20-H200, 1x8, SLURM `1576160`, `1581240` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_750M_50B_job1576160) | `$OUT/tsz128x4k_50B_repro_ngocbh_kda_750M_50B_job1576160/final-model-ckpt.pth` |
| 750M / 50B | Isotropic KDN | `repro_ngocbh_iso_kla_750M_50B_job1576161` | T20-H200, 1x8, SLURM `1576161`, `1576803`, `1588990`, `1592097` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_750M_50B_job1576161) | `$OUT/tsz128x4k_50B_repro_ngocbh_iso_kla_750M_50B_job1576161/final-model-ckpt.pth` |
| 750M / 50B | Diagonal KDN | `repro_ngocbh_diag_kla_750M_50B_isdk` | MAST-H100, 8x8, `hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_isdk); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/` | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/final-model-ckpt.pth` |
| 1.3B / 100B | GDN-2 | `repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697856` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697857` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | KDA | `repro_ngocbh_kda_1.3B_100B_bs0p5m` | MAST-H100, 8x8, `hdn_kda_1p3B_100B-ngocbh-grw6jgqn` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_kda_1p3B_100B-ngocbh-grw6jgqn); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | Isotropic KDN | `repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B | Diagonal KDN | `repro_ngocbh_diag_kla_1.3B_100B_bs0p5m` | MAST-H100, 16x8, `hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | GDN-2 + SWA | `repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645030` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | Mamba-3 (SISO) + SWA | `repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1722329` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | KDA + SWA | `repro_ngocbh_swa_kda_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1677268` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_kda_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | Isotropic KDN + SWA | `repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645029` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |

All listed local checkpoints were verified on 2026-08-21. For MAST rows, the
listed `OUT` file is the local mirror; the native artifact has the same suffix
under `$MF/outputs/`. W&B on those rows contains uploaded evaluation results,
while the linked MAST/TensorBoard run is the training record.

The Isotropic KDN files were losslessly migrated in place to the packed
production layout. Their pre-migration backups are under
`/checkpoints/ngocbh/hdn/ckpts/isokla_checkpoint_backups/2026-08-12-packed-production/`.
