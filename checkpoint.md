# Paper checkpoints

Source of truth for the trained checkpoints used by the result tables in
`main_arxiv.tex`. The retrieval table reuses the 750M and 1.3B checkpoints
below. The information-scale table uses three matched 750M Diagonal KDN arms
with random omega initialization, the main-table $\mu=d_k$ checkpoint, and one
learned-scale checkpoint initialized at $d_k$; the overwrite appendix additionally
uses the matched random-gain $\mu=d_k$ arm. The fixed-noise table uses two matched
750M Diagonal KDN arms; the throughput figure does not use a checkpoint. Rows labeled
``held-out registry only'' are superseded table checkpoints retained because the
local held-out FineWeb manifest still references their verified mirror; they are not
used by the current result tables.

Path and cluster abbreviations:

- `OUT=/checkpoints/ngocbh/hdn/ckpts/outputs`
- `MF=manifold://deep_retrieval/tree/datasets/ngocbh_hdn/ckpts`
- `T20-H200=shared-aws-usw1-1`; `MAST-H100=grandteton`; `MAST-B200=grandteton_b200`
- GPU allocations are written as nodes x GPUs per node.

Historical IsoKLA checkpoints predate the current $q_{\min}=0$,
$r_{\min}=0.01$ defaults. Reconstruct the 220M and 750M checkpoints with
`iso_kla_220M_legacy_q0_r0p1` and `iso_kla_750M_legacy_q0_r0p1`, respectively;
use `iso_kla_1.3B_legacy_q0p05_r0p05` and
`swa_iso_kla_1.3B_legacy_q0p05_r0p05` for the recurrent and hybrid 1.3B checkpoints.

| Paper block | Model | Run ID | Training cluster / job | Tracker | Final checkpoint |
|---|---|---|---|---|---|
| 220M / 20B | DeltaNet | `repro_ngocbh_deltanet_220M_20B_0718_0136` | T20-H200, 1x8, SLURM `1539011` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_deltanet_220M_20B_0718_0136) | `$OUT/tsz128x4k_20B_repro_ngocbh_deltanet_220M_20B_0718_0136/final-model-ckpt.pth` |
| 220M / 20B | Gated DeltaNet | `repro_ngocbh_gateddeltanet_220M_20B_0718_0131` | T20-H200, 1x8, SLURM `1538989` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gateddeltanet_220M_20B_0718_0131) | `$OUT/tsz128x4k_20B_repro_ngocbh_gateddeltanet_220M_20B_0718_0131/final-model-ckpt.pth` |
| 220M / 20B | GDN-2 | `repro_ngocbh_gdn2_220M_20B_0714_0317` | T20-H200, 1x8, SLURM `1521127` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_220M_20B_0714_0317) | `$OUT/tsz128x4k_20B_repro_ngocbh_gdn2_220M_20B_0714_0317/final-model-ckpt.pth` |
| 220M / 20B | KDA | `repro_ngocbh_kda_220M_20B_0718_0338` | T20-H200, 1x8, SLURM `1539391` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_220M_20B_0718_0338) | `$OUT/tsz128x4k_20B_repro_ngocbh_kda_220M_20B_0718_0338/final-model-ckpt.pth` |
| 220M / 20B | Isotropic KDN | `repro_ngocbh_iso_kla_220M_20B_0724_1833` | T20-H200, 1x8, SLURM `1564904`, `1565202` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_220M_20B_0724_1833) | `$OUT/tsz128x4k_20B_repro_ngocbh_iso_kla_220M_20B_0724_1833/final-model-ckpt.pth` |
| 220M / 20B | Diagonal KDN | `repro_ngocbh_diag_kla_220M_20B_isdk` | MAST-B200, 2x8, `hdn_diag_kla_220M_20B_isdk-ngocbh-bbhmhg54` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_220M_20B_isdk-ngocbh-bbhmhg54); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_220M_20B_isdk); TB: `$MF/tb/tsz128x4k_20B_repro_ngocbh_diag_kla_220M_20B_isdk/` | `$OUT/tsz128x4k_20B_repro_ngocbh_diag_kla_220M_20B_isdk/final-model-ckpt.pth` |
| 750M / 50B | GDN-2 | `repro_ngocbh_gdn2_750M_50B_job1576159` | T20-H200, 1x8, SLURM `1576159`, `1580790` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_750M_50B_job1576159) | `$OUT/tsz128x4k_50B_repro_ngocbh_gdn2_750M_50B_job1576159/final-model-ckpt.pth` |
| 750M / 50B | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1693970` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_siso_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1694025` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_mimo_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | KDA | `repro_ngocbh_kda_750M_50B_job1576160` | T20-H200, 1x8, SLURM `1576160`, `1581240` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_750M_50B_job1576160) | `$OUT/tsz128x4k_50B_repro_ngocbh_kda_750M_50B_job1576160/final-model-ckpt.pth` |
| 750M / 50B | Isotropic KDN | `repro_ngocbh_iso_kla_750M_50B_job1576161` | T20-H200, 1x8, SLURM `1576161`, `1576803`, `1588990`, `1592097` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_750M_50B_job1576161) | `$OUT/tsz128x4k_50B_repro_ngocbh_iso_kla_750M_50B_job1576161/final-model-ckpt.pth` |
| 750M / 50B | Diagonal KDN | `repro_ngocbh_diag_kla_750M_50B_isdk` | MAST-H100, 8x8, `hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_isdk); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/` | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=1$) | `repro_ngocbh_diag_kla_750M_isone_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_isone_randgain-ngocbh-nkgxcj5r` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_isone_randgain-ngocbh-nkgxcj5r); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_isone_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_isone_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=\sqrt{d_k}$) | `repro_ngocbh_diag_kla_750M_issqrt_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_issqrt_randgain-ngocbh-zjtfqr2w` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_issqrt_randgain-ngocbh-zjtfqr2w); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_issqrt_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_issqrt_randgain_50B/final-model-ckpt.pth` |
| Overwrite diagnostic, 750M / 50B | Diagonal KDN ($\mu=d_k$, random gain) | `repro_ngocbh_diag_kla_750M_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_randgain-ngocbh-phm3fzd9` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_randgain-ngocbh-phm3fzd9); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=4d_k$) | `repro_ngocbh_diag_kla_750M_is4dk_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_is4dk_randgain-ngocbh-z96zvtlq` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_is4dk_randgain-ngocbh-z96zvtlq); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_is4dk_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_is4dk_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN (learned $\mu$, init. $d_k$) | `repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk` | T20-H200, 2x8, SLURM `1711368` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk/final-model-ckpt.pth` |
| Noise ablation, 750M / 50B | Diagonal KDN ($r_t=1$, learned $\omega_t$) | `repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1` | T20-H200, 2x8, SLURM `1725572` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1/final-model-ckpt.pth` |
| Noise ablation, 750M / 50B | Diagonal KDN ($r_t=1$, $\omega_t=1$) | `repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1` | T20-H200, 2x8, SLURM `1725635` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1/final-model-ckpt.pth` |
| 1.3B / 100B | GDN-2 | `repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697856` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697857` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | KDA | `repro_ngocbh_kda_1.3B_100B_bs0p5m` | MAST-H100, 8x8, `hdn_kda_1p3B_100B-ngocbh-grw6jgqn` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_kda_1p3B_100B-ngocbh-grw6jgqn); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B | Isotropic KDN | `repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B | Diagonal KDN (learned per-head information scale and initial precision $c_0$) | `repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200` | MAST-B200, 16x8, `fire-ngocbh-UFM-hdn-d1p3b-100b-s3-learnis-lmu-iso-q1em6-r0p-pjx093p4` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/fire-ngocbh-UFM-hdn-d1p3b-100b-s3-learnis-lmu-iso-q1em6-r0p-pjx093p4); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200/` | `$MF/outputs/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200/final-model-ckpt.pth` |
| Held-out registry only (superseded 1.3B / 100B) | Diagonal KDN | `repro_ngocbh_diag_kla_1.3B_100B_bs0p5m` | MAST-H100, 16x8, `hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | GDN-2 + SWA | `repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645030` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | Mamba-3 (SISO) + SWA | `repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1722329` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | Mamba-3 (MIMO) + SWA | `repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1722330` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | KDA + SWA | `repro_ngocbh_swa_kda_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1677268` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_kda_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B | Isotropic KDN + SWA | `repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645029` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |

All listed local checkpoints were verified by 2026-08-24. For MAST rows that
use `OUT`, the listed file is the local mirror and the native artifact has the
same suffix under `$MF/outputs/`; rows using `MF` point directly to the native
artifact. W&B on rows that link it contains uploaded evaluation results, while
the linked MAST/TensorBoard run is the training record.

The Isotropic KDN files were losslessly migrated in place to the packed
production layout. Their pre-migration backups are under
`/checkpoints/ngocbh/hdn/ckpts/isokla_checkpoint_backups/2026-08-12-packed-production/`.
