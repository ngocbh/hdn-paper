# Paper checkpoints

Source of truth for the trained checkpoints used by the result tables in
`main_arxiv.tex`. The retrieval table reuses the 750M and 1.3B checkpoints
below. The information-scale table uses three matched 750M Diagonal KDN arms
with random omega initialization, the main-table $\mu=d_k$ checkpoint, and one
learned-scale checkpoint initialized at $d_k$; the overwrite appendix additionally
uses the matched random-gain $\mu=d_k$ arm. The fixed-noise table uses two matched
750M Diagonal KDN arms; the throughput figure does not use a checkpoint. Rows labeled
"held-out registry only" are superseded table checkpoints retained because the
local held-out FineWeb manifest still references their verified mirror; they are not
used by the current result tables. The overwrite-diagnostic checkpoint is likewise
excluded from that held-out manifest because it supports only the diagnostic appendix.
A table checkpoint whose final artifact is listed directly under `$MF/outputs/` is
native-only and joins the held-out manifest after a local mirror and hash are verified.

Path and cluster abbreviations:

- `OUT=/checkpoints/ngocbh/hdn/ckpts/outputs`
- `MF=manifold://deep_retrieval/tree/datasets/ngocbh_hdn/ckpts`
- `T20-H200=shared-aws-usw1-1`; `MAST-H100=grandteton`; `MAST-B200=grandteton_b200`
- GPU allocations are written as nodes x GPUs per node.

Historical IsoKLA checkpoints predate the current $q_{\min}=0$,
$r_{\min}=0.01$ defaults. Reconstruct the 750M checkpoint with
`iso_kla_750M_legacy_q0_r0p1`;
use `iso_kla_1.3B_legacy_q0p05_r0p05` and
`swa_iso_kla_1.3B_legacy_q0p05_r0p05` for the historical recurrent and hybrid
1.3B checkpoints, respectively. The newer hybrid $q_{\min}=0$,
$r_{\min}=0.01$ checkpoint uses `swa_iso_kla_1.3B_q0_r0p01`.

| Paper block | Model | Run ID | Training cluster / job | Tracker | Final checkpoint |
|---|---|---|---|---|---|
| 750M / 50B | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1693970` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_siso_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_750M_50B_decayfix` | T20-H200, 2x8, SLURM `1694025` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_750M_50B_decayfix) | `$OUT/tsz128x4k_50B_repro_ngocbh_mamba3_mimo_750M_50B_decayfix/final-model-ckpt.pth` |
| 750M / 50B | KDA | `repro_ngocbh_kda_750M_50B_job1576160` | T20-H200, 1x8, SLURM `1576160`, `1581240` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_750M_50B_job1576160) | `$OUT/tsz128x4k_50B_repro_ngocbh_kda_750M_50B_job1576160/final-model-ckpt.pth` |
| 750M / 50B | GDN-2 | `repro_ngocbh_gdn2_750M_50B_job1576159` | T20-H200, 1x8, SLURM `1576159`, `1580790` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_750M_50B_job1576159) | `$OUT/tsz128x4k_50B_repro_ngocbh_gdn2_750M_50B_job1576159/final-model-ckpt.pth` |
| 750M / 50B | Isotropic KDN | `repro_ngocbh_iso_kla_750M_50B_job1576161` | T20-H200, 1x8, SLURM `1576161`, `1576803`, `1588990`, `1592097` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_750M_50B_job1576161) | `$OUT/tsz128x4k_50B_repro_ngocbh_iso_kla_750M_50B_job1576161/final-model-ckpt.pth` |
| 750M / 50B | Diagonal KDN | `repro_ngocbh_diag_kla_750M_50B_isdk` | MAST-H100, 8x8, `hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_isdk-ngocbh-c2gxchdd); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_isdk); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/` | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_isdk/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=1$) | `repro_ngocbh_diag_kla_750M_isone_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_isone_randgain-ngocbh-nkgxcj5r` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_isone_randgain-ngocbh-nkgxcj5r); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_isone_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_isone_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=\sqrt{d_k}$) | `repro_ngocbh_diag_kla_750M_issqrt_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_issqrt_randgain-ngocbh-zjtfqr2w` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_issqrt_randgain-ngocbh-zjtfqr2w); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_issqrt_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_issqrt_randgain_50B/final-model-ckpt.pth` |
| Overwrite diagnostic, 750M / 50B | Diagonal KDN ($\mu=d_k$, random gain) | `repro_ngocbh_diag_kla_750M_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_randgain-ngocbh-phm3fzd9` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_randgain-ngocbh-phm3fzd9); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN ($\mu=4d_k$) | `repro_ngocbh_diag_kla_750M_is4dk_randgain_50B` | MAST-B200, 8x8, `hdn_diag_kla_750M_50B_is4dk_randgain-ngocbh-z96zvtlq` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_750M_50B_is4dk_randgain-ngocbh-z96zvtlq); TB: `$MF/tb/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_is4dk_randgain_50B/` | `$OUT/diag_kla_ablations/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_is4dk_randgain_50B/final-model-ckpt.pth` |
| $\mu$ ablation, 750M / 50B | Diagonal KDN (learned $\mu$, init. $d_k$) | `repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk` | T20-H200, 2x8, SLURM `1711368` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_lscale_fixedmu_initdk/final-model-ckpt.pth` |
| Noise ablation, 750M / 50B | Diagonal KDN ($r_t=1$, learned $\omega_t$) | `repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1` | T20-H200, 2x8, SLURM `1725572` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1/final-model-ckpt.pth` |
| Noise ablation, 750M / 50B | Diagonal KDN ($r_t=1$, $\omega_t=1$) | `repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1` | T20-H200, 2x8, SLURM `1725635` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1) | `$OUT/tsz128x4k_50B_repro_ngocbh_diag_kla_750M_50B_chunk_kalman_s3_scale1_fixedmu_fixedr1_fixedomega1/final-model-ckpt.pth` |
| 1.3B / 100B (Tables 1--3) | Mamba-3 (SISO) | `repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697856` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B (Tables 1--3) | Mamba-3 (MIMO) | `repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1697857` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_mamba3_mimo_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B (Tables 1--3) | KDA | `repro_ngocbh_kda_1.3B_100B_bs0p5m` | MAST-H100, 8x8, `hdn_kda_1p3B_100B-ngocbh-grw6jgqn` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_kda_1p3B_100B-ngocbh-grw6jgqn); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_kda_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B (Tables 1--3) | GDN-2 | `repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_gdn2_1p3B_100B_b200-ngocbh-vlftqd5v); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_gdn2_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B (Tables 1--3) | Isotropic KDN | `repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200` | MAST-B200, 8x8, `hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_iso_kla_1p3B_100B-ngocbh-hlfvpdn5); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/` | `$OUT/tsz128x4k_100B_repro_ngocbh_iso_kla_1.3B_100B_bs0p5m_b200/final-model-ckpt.pth` |
| 1.3B / 100B | Diagonal KDN (RULER Table 3, learned per-head information scale and initial precision $p_0^{-1}$) | `repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200` | MAST-B200, 16x8, `fire-ngocbh-UFM-hdn-d1p3b-100b-s3-learnis-lmu-iso-q1em6-r0p-pjx093p4` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/fire-ngocbh-UFM-hdn-d1p3b-100b-s3-learnis-lmu-iso-q1em6-r0p-pjx093p4); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200/` | `$MF/outputs/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_learnis_learnmu_isoinit_q1em6_r0p01_100B_s3_16n_b200/final-model-ckpt.pth` |
| Held-out registry only (superseded 1.3B / 100B) | Diagonal KDN | `repro_ngocbh_diag_kla_1.3B_100B_bs0p5m` | MAST-H100, 16x8, `hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g` | [MAST](https://www.internalfb.com/mlhub/pipelines/runs/mast/hdn_diag_kla_1p3B_100B-ngocbh-hz0pcm3g); [W&B eval](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_1.3B_100B_bs0p5m); TB: `$MF/tb/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/` | `$OUT/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| 1.3B / 100B language modeling and real-world retrieval (Tables 1--2) | Diagonal KDN (fixed information scale $\mu=d_k$; learned per-head initial precision $p_0^{-1}$; $q_{\min}=0$, $r_{\min}=0.01$) | `repro_ngocbh_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m` | T20-H200, 2x8, SLURM `1773604` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m/final-model-ckpt.pth` |
| Information-scale study, 1.3B / 100B | Diagonal KDN (learned information scale $\mu$, initialized at $d_k$; learned initial precision $p_0^{-1}$; $q_{\min}=0$, $r_{\min}=0.01$) | `repro_ngocbh_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m` | T20-H200, 2x8, SLURM `1773605` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid / attention-only 1.3B / 100B (Tables 1--3) | SWA Transformer (2K window) | `repro_ngocbh_swa_transformer_1.3B_2k_100B_bs0p5m` | T20-H200, 2x8, SLURM `1776169` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_transformer_1.3B_2k_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_transformer_1.3B_2k_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Tables 1--3) | Mamba-3 (SISO) + SWA | `repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1722329` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_mamba3_siso_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Tables 1--3) | Mamba-3 (MIMO) + SWA | `repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1722330` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_mamba3_mimo_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Tables 1--3) | KDA + SWA | `repro_ngocbh_swa_kda_1.3B_100B_bs0p5m` | T20-H200, 2x8, SLURM `1677268` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_kda_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_kda_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Tables 1--3) | GDN-2 + SWA | `repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645030` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_gdn2_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Tables 1--3) | Isotropic KDN + SWA | `repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m` | T20-H200, 4x8, SLURM `1645029` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_iso_kla_1.3B_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B (Table 1) | Diagonal KDN + SWA (fixed information scale $\mu=d_k$; learned per-head initial precision $p_0^{-1}$; $q_{\min}=0$, $r_{\min}=0.01$) | `repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m` | T20-H200, 2x8, SLURM `1773606` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid 1.3B / 100B real-world and RULER retrieval (Tables 2--3) | Diagonal KDN + SWA (fixed information scale $\mu=d_k$; learned per-head initial precision $p_0^{-1}$; $q_{\min}=10^{-6}$, $r_{\min}=0.01$) | `repro_ngocbh_swa_diag_kla_1.3B_q1em6_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m` | T20-H200, 4x8, SLURM `1806211` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_diag_kla_1.3B_q1em6_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_diag_kla_1.3B_q1em6_r0p01_isdk_fixedscale_learnmu_isoinit_100B_bs0p5m/final-model-ckpt.pth` |
| Hybrid information-scale study, 1.3B / 100B | Diagonal KDN + SWA (learned information scale $\mu$, initialized at $d_k$; learned initial precision $p_0^{-1}$; $q_{\min}=0$, $r_{\min}=0.01$) | `repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m` | T20-H200, 2x8, SLURM `1773607` | [W&B](https://wandb.ai/ngocjr7/llm_next_gen/runs/repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m) | `$OUT/tsz128x4k_100B_repro_ngocbh_swa_diag_kla_1.3B_q0_r0p01_learnis_dkinit_learnmu_isoinit_100B_bs0p5m/final-model-ckpt.pth` |

## Real-world retrieval evaluation

The real-world retrieval table uses protocol
`mamba3_gdn2_table4_paper_compat_v1`, ported from
`HazyResearch/prefix-linear-attention` commit
`7490f22bda3e38dc057bcbaffe6bdb09b4d475e6`. The pinned evaluator runtime is
commit `e3bc0bde5c79f658b3f28d427a31884dbf8ba1d3` with source SHA-256
`21ab5af6e6771b6b2bdcac87dedc0897d22253516b045df49ee0a516e431d9c1`;
the final aggregation/upload tooling is commit
`ab797368a53452303c307400fd43680c5117636a` with source SHA-256
`a897d1843a0c632a6660a3884cc486b81ae298b00a950af88a66a770b83b5023`.
The checkpoint manifest SHA-256 is
`23bb51928e123a40725a21279843515b439fcbe57fe26f1ba77ae319ccdce579`
and the dataset manifest SHA-256 is
`2d1f1109561f87dac561e6688a4b1a7a2b6c79109f70120b8f9fea362b5fcd52`.
Each checkpoint evaluates 12,077 documents in BF16 with fixed batch size 8,
right padding to 2,048 tokens, no padding mask, full-prefix recomputation, and
the selected-token LM head. Full reports are stored as
`<checkpoint-dir>/retrieval_table4_eval/report.json`.

The main-paper rows are manifest indices 13--18, 20--25, and 27. Reference
jobs `1818896_16` and `1818896_24` plus sweep array `1820186` produced all 29
local reports. The sealed aggregate is
`/storage/home/ngocbh/datasets/retrieval_table4_v1/results/full_1820186_ab79736`:
`summary.json` SHA-256
`899527ed1962a1d3776f1edfca2d90ce154f52bedcac63314802c452ec9674c5`,
`table.csv` `bc08aefee71f77aca343df04fd38ba1ccfb2b45aec8c4ed84a69f3508b89208d`,
`table.md` `71fc04a0f3bf1f7c8dfe779078105cf844f8b8ac7732b253e8416177364ca890`,
and `table.tex` `69c7213da172ce488bcc49153ae5d00346b29b9bf06dceb1769e86884194c3ab`.
W&B preflight/upload/verification jobs were `1823498`, `1823510`, and
`1823541`; the final verification classified all 29 targets as identical-hash
skips. Metrics use the `eval/retrieval_table4/*` namespace. The Manifold-only
learned-scale recurrent Diagonal KDN checkpoint is explicitly excluded because
it cannot be read from this cluster; it is not silently replaced by another
checkpoint.

All listed local checkpoints were verified by 2026-09-02. For MAST rows, the
listed `OUT` file is the local mirror; the native artifact has the same suffix
under `$MF/outputs/`. W&B on those rows contains uploaded evaluation results,
while the linked MAST/TensorBoard run is the training record.

SLURM jobs `1773238`, `1773604`–`1773607`, and `1776169` reached natural
training completion and wrote the listed final checkpoints. Their top-level
`FAILED` states came from post-training evaluation or handoff failures, not
from training failure.

The Isotropic KDN files were losslessly migrated in place to the packed
production layout. Their pre-migration backups are under
`/checkpoints/ngocbh/hdn/ckpts/isokla_checkpoint_backups/2026-08-12-packed-production/`.
