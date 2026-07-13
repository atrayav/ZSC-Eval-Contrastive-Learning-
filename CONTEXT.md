# Project Context — ZSC-Eval Contrastive Partner Encoder

> **Purpose of this file:** a single place to get anyone (you, a collaborator, or an
> AI assistant in a fresh session) up to speed fast. Read this top-to-bottom and you
> should be able to continue the work without re-deriving anything.
>
> Last updated: 2026-07-12.

---

## 1. The research in one paragraph

We are extending **ZSC-Eval** (NeurIPS 2024, `sjtu-marl/ZSC-Eval`) to test a claim:
*a policy conditioned on an inferred **partner embedding** should coordinate better with
held-out partners than population-only baselines (SP / FCP / MEP).* The embedding is
produced by a small **contrastive partner encoder** — a GRU that watches a partner's
recent observations and outputs a 32-D vector trained with an InfoNCE loss. Success is
measured with **BR-Prox** on Overcooked layouts, primarily `random3` (hardest) and
`unident_s` (cleanest asymmetry).

## 2. Current status

- **Staged pipeline complete and gated** (07-12): offline pre-training on FCP rollouts
  passed the go/no-go probe — 30.9% held-out partner id vs 11.1% chance. Deploy the
  **3k-step** checkpoint (`data/partner_windows/encoder_random3.pt`); 10k overfits.
  Finding: the embedding mostly encodes partner *competence* (71% at init/mid/final
  stage) rather than individual identity — frame the write-up accordingly.
- **FCP Stage 2 conditioning wired and smoke-tested** end-to-end (07-12).
- **READY TO RUN:** the full-scale conditioned run (see §5). Not yet started —
  ~7 days/seed on the laptop at 12 threads; Hyak access pending.
- Baselines already reproduced (see `RESEARCH_LOG.md`): e.g. `random3` SP=0.10, FCP=0.64, MEP=0.70.
  **Number to beat: FCP BR-Prox 0.635 on `random3`.**

## 3. Where the code lives (important: two checkouts)

| Location | Role |
|---|---|
| `~/ZSC-Eval` (WSL Ubuntu, i.e. `/home/varun/ZSC-Eval`) | **Authoritative.** Runs training, holds git, pushes to GitHub. |
| `C:\Users\varun\ZSC-Eval` (Windows) | Editing convenience. Has **drifted** from the fork — do not assume it matches. |
| `https://github.com/atrayav/ZSC-Eval-Contrastive-Learning-` | Remote. **Standalone repo, not a fork**, so commits count toward contributions. |

**Golden rule:** git operations happen in WSL. To edit, either sync a file down first
(`cp ~/ZSC-Eval/<f> /mnt/c/.../ZSC-Eval/<f>`) or edit on Windows then sync up
(`cp /mnt/c/.../ZSC-Eval/<f> ~/ZSC-Eval/<f>`) and commit in WSL.

## 4. The files we added / changed

| File | What it does in this feature |
|---|---|
| `zsceval/algorithms/r_mappo/algorithm/contrastive_encoder.py` | **NEW.** `PartnerEncoder` (GRU→32-D, L2-normalized) + `infonce_loss` (symmetric NT-Xent). |
| `zsceval/algorithms/r_mappo/algorithm/r_actor_critic.py` | Actor concatenates the embedding to its features before the action head (`_cat_partner_emb`). |
| `zsceval/algorithms/r_mappo/algorithm/rMAPPOPolicy.py` | Builds the encoder + its own optimizer; threads `partner_emb` through act/evaluate. |
| `zsceval/algorithms/r_mappo/r_mappo.py` | `update_encoder()` (InfoNCE training) + `ppo_update()` uses the stored embedding. |
| `zsceval/utils/shared_buffer.py` | Stores `partner_embs`; all 3 generators yield it as a 14th field. |
| `zsceval/runner/shared/overcooked_runner.py` | Rolling partner-obs window, `_get_partner_emb`, eval window, encoder checkpointing. |
| `zsceval/overcooked_config.py` | 8 CLI args (see below). |
| `zsceval/scripts/overcooked/shell/train_contrastive_sp.sh` | **NEW.** SP launch script (legacy path). |
| `zsceval/scripts/overcooked/collect_partner_windows.py` | **NEW.** Rollouts vs each frozen FCP S1 partner → labeled obs windows. |
| `zsceval/scripts/overcooked/pretrain_partner_encoder.py` | **NEW.** Offline InfoNCE pre-training + k-NN probe gate. |
| `zsceval/algorithms/population/trainer_pool.py` | Partner windows + frozen-encoder embeddings in the population (S2) loop. |
| `zsceval/scripts/overcooked/train/train_adaptive.py` | Injects encoder flags; loads + freezes the pre-trained encoder. |
| `zsceval/scripts/overcooked/shell/train_fcp_s2_contrastive.sh` | **NEW.** THE full-scale launch script (treatment + baseline modes). |
| `zsceval/scripts/overcooked/shell/train_fcp_s2_contrastive_smoke.sh` | **NEW.** Minutes-long wiring check for the above. |

## 5. How to run it

**THE run to do next (staged setup, all in WSL):**

```bash
cd ~/ZSC-Eval/zsceval/scripts/overcooked/shell
# treatment: adaptive agent conditioned on the frozen pre-trained encoder
bash train_fcp_s2_contrastive.sh random3 12
# control: identical run, no encoder (same-codebase FCP baseline)
bash train_fcp_s2_contrastive.sh random3 12 baseline
# quick wiring check first if anything changed (a few minutes):
bash train_fcp_s2_contrastive_smoke.sh
```

Env overrides: `SEED_BEGIN`/`SEED_MAX` (default 1..5), `N_THREADS` (default 100 —
use ~12 on the laptop), `ENCODER_CKPT`. Logs tee to `~/ZSC-Eval/logs/`.
On a fresh machine first regenerate the gitignored inputs: population yamls
(`cd zsceval/scripts && python prep/gen_S2_yml.py random3 fcp`) and the encoder
checkpoint (`collect_partner_windows.py` then `pretrain_partner_encoder.py`,
run from `zsceval/scripts/overcooked/`, or copy `data/partner_windows/` over).

**Legacy SP path** (superseded by the staged setup, kept for ablations):
`train_contrastive_sp.sh random3 [--condition]` — `--use_partner_encoder` alone
learns hunches jointly during SP; `--condition_actor_on_partner` also acts on them.

## 6. ✅ The SP-negatives problem — RESOLVED (2026-07-12)

The original wiring trained the encoder on **self-play** rollouts, where every thread
runs the *same* policy, so InfoNCE's "negatives" were not distinct partners.

**Resolution: Option 2 — staged pre-training** (see RESEARCH_LOG 2026-07-12 entries).
The encoder is pre-trained offline on rollouts against the 45 frozen FCP Stage-1
partners (`collect_partner_windows.py` → `pretrain_partner_encoder.py`), gated by a
k-NN probe on held-out partners (passed), then loaded FROZEN into FCP Stage 2 via
`--pretrained_encoder_path`. Key implementation invariant: observations are divided
by `partner_obs_scale` (255, stored in the checkpoint) before encoding — raw
image-scaled obs saturate the GRU. The next open question is empirical, not
architectural: does conditioning beat FCP 0.635 BR-Prox on `random3`?

## 7. Gotchas that cost time before

- **Git identity:** commits must be authored `varun.atraya12@gmail.com` (the GitHub-verified
  email) or they don't count as contributions. Global config is set correctly now; if green
  squares stop, re-check `git config user.email`.
- **Overcooked obs are 3-D** (e.g. 8×5×20) — anything feeding the encoder must flatten with
  `int(np.prod(obs_shape))`.
- **PPO log-prob match:** the embedding used when acting MUST be the same one fed during
  `evaluate_actions`, or the PPO ratio is wrong. This is why embeddings are stored in the buffer.
- **Nested quoting** through PowerShell→WSL→bash breaks constantly — write a `.sh` file and run it.
- **`dummy_batch_size` must divide `n_rollout_threads`** (e.g. 10 threads / batch 2).

## 8. Paper lineage (for the write-up)

- **PEARL** (Rakelly et al., ICML 2019): the "encode context → latent → condition policy"
  skeleton we follow, with InfoNCE swapped in for PEARL's VAE/ELBO objective.
- **CACL** (arXiv 2307.01403): contrastive learning in MARL (communication) — the MARL-contrastive cite.
- Neither is a direct reproduction; the specific combination for ZSC partner inference is the contribution.

## 9. Pointers

- `RESEARCH_LOG.md` — dated progress entries and baseline tables.
- `README.md` — public-facing summary + baseline table.
- Failure-mode checks to run once training works: probe accuracy on held-out partners,
  constant-(zero)-embedding ablation, within-episode adaptation curve, ZSC score on held-out population.
