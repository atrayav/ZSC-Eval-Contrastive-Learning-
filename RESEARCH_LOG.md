# Research Log — ZSC-Eval Contrastive Learning

## Research Goal

Test whether contrastively learned partner representations improve zero-shot coordination (ZSC) over population-only baselines in Overcooked multi-agent RL.

**Core claim to validate:** A policy conditioned on an inferred partner embedding should coordinate better with held-out partners than a single population-trained generalist policy.

---

## 2026-06-22 — Setup & First Baseline

### What we did
- Cloned `sjtu-marl/ZSC-Eval` locally to `C:\Users\varun\ZSC-Eval`
- Identified WSL2 as the correct runtime (repo is bash/conda-first, not Windows-native)
- Found existing `~/miniconda3` in WSL Ubuntu — created `zsceval` conda env from `environment.yml` (Python 3.9, PyTorch 2.5.1, CUDA 11.8)
- Cloned pretrained policy pool from HuggingFace (`Leoxxxxh/ZSC-Eval-policy_pool`) into `zsceval/policy_pool/`
- Fixed CRLF line endings on all `.sh` scripts (Windows→Linux issue)
- Fixed editable package install to point to native WSL path (`~/ZSC-Eval`) instead of `/mnt/c/` to avoid binary pickle corruption
- Ran first clean FCP evaluation on `random0`

### Issues encountered
- `.sh` scripts had Windows `\r\n` line endings — fixed with `sed -i 's/\r//' *.sh`
- `pip install -e` pointed to `/mnt/c/` path causing `_pickle.UnpicklingError` on planner files — fixed by reinstalling from `~/ZSC-Eval`
- `unident_s` OOM-killed at default 400 threads — fixed by reducing to 40 threads

### Baseline results — `random0` (BR-Prox, IQM across 10 partners)

| Algorithm | pos0 | pos1 | Overall |
|---|---|---|---|
| SP | 0.998 | 0.699 | 0.857 |
| FCP s24 | 0.992 | 0.737 | 0.873 |
| FCP s36 | 0.993 | 0.720 | 0.877 |
| MEP s12 | 0.990 | 0.742 | 0.870 |
| MEP s36 | 0.992 | 0.704 | 0.873 |

---

## 2026-06-24 — Full Baseline Across All Layouts

### What we did
- Ran SP, FCP, MEP evaluations across all 6 old Overcooked layouts
- Extracted BR-Prox scores for each
- Pushed results to fork: `https://github.com/atrayav/ZSC-Eval-Contrastive-Learning-`

### Full baseline table (best variant per algo, overall BR-Prox)

| Layout | SP | FCP (best) | MEP (best) |
|---|---|---|---|
| `random0` | 0.857 | 0.877 | 0.873 |
| `random0_medium` | 0.567 | 0.643 | 0.586 |
| `random1` | 0.274 | 0.708 | 0.763 |
| `random3` | 0.100 | 0.635 | 0.696 |
| `small_corridor` | 0.191 | 0.553 | 0.876 |
| `unident_s` | 0.469 | 0.967 | 0.965 |

### Key observations
- `random0` shows near-saturation — SP nearly matches FCP/MEP, not a useful discriminative benchmark
- `random3` is the hardest layout — SP collapses to 0.10, largest gap for a contrastive method to close
- `unident_s` has the cleanest separation: SP=0.47 vs FCP/MEP=0.97, strong pos0/pos1 asymmetry
- `small_corridor` FCP shows unusual variance across population sizes (s12=0.54, s24=0.26, s36=0.55) — worth investigating
- All results reproduced cleanly from pretrained policy pool, consistent with paper expectations

### Notable flag
`unident_s` was evaluated with 400 episodes (vs default 800) due to laptop memory constraints. Scores are plausible but have slightly higher variance than other layouts.

### Best target layouts for contrastive work
- **`random3`** — largest SP-to-population gap, hardest coordination challenge
- **`unident_s`** — cleanest asymmetry, best separation between weak and strong algorithms

---

---

## 2026-06-26 — Contrastive Implementation Complete

### What we did
- Implemented full contrastive partner encoder stack across 7 files:
  - `contrastive_encoder.py` — GRU encoder + InfoNCE loss
  - `r_actor_critic.py` — actor accepts and concatenates partner embedding
  - `rMAPPOPolicy.py` — instantiates encoder, threads embedding through get_actions/act/evaluate_actions
  - `r_mappo.py` — update_encoder() with anchor/positive episode-split InfoNCE; ppo_update() unpacks 14-field sample
  - `overcooked_config.py` — 8 new args (use_partner_encoder, partner_emb_dim, encoder_context_len, encoder_hidden_size, infonce_temperature, encoder_lr, infonce_coef, condition_actor_on_partner)
  - `overcooked_runner.py` — rolling observation window in collect/eval; encoder checkpointing
  - `shared_buffer.py` — partner_embs storage; all 3 generators yield 14-field samples
- Fixed PPO log-prob mismatch: embeddings used at collection time now stored in buffer and passed to evaluate_actions()
- Fixed len(sample)=13 crash path with defensive fallback in ppo_update()
- Fixed eval() always being called on last episode regardless of use_eval flag
- Fixed encoder checkpoint restore path mismatch
- Dry-run passed cleanly on random3 (8000 steps, no errors)
- Pushed to GitHub: `https://github.com/atrayav/ZSC-Eval-Contrastive-Learning-`

### Key method caveat identified
Current positive/negative construction uses the **same SP policy across all rollout threads**: anchor = first-half trajectory, positive = second-half trajectory, negatives = other threads. Since all threads run the same policy, InfoNCE may learn to distinguish trajectory segments or environment states rather than partner identity. This undermines the core claim.

### Papers identified as relevant
- arXiv:2307.01403 — "Learning Multi-Agent Communication with Contrastive Learning" — closest structural match for InfoNCE on MARL trajectory views
- OpenReview LWmuPfEYhH — "Attention-Guided Contrastive Role Representations for Multi-agent" — contrastive role/type inference in MARL
- arXiv:2209.15618 — "Beyond Bayes-optimality" — theoretical motivation for why expected-return agents fail under partner uncertainty

## 2026-07-08 — Documentation Pass Over the Encoder Stack

### What we did
- Commented the entire contrastive stack in 26 single-concern commits — every non-obvious decision now has an explanation at the code site
- Highlights: why the stored embedding must feed the PPO ratio (log-prob match), why eval must mirror collection-time embeddings, the rolling-window FIFO slide and zero-init cold start, the anchor/positive episode split and its self-play caveat, the 13-field backward-compat fallback in `ppo_update()`, and the dual encoder save that keeps save/restore paths in sync

### Why
- The June implementation carried implicit invariants discovered during debugging; encoding them as comments at the code site makes the repo safe to hand off and hard to break silently

---


## 2026-07-10 — CONTEXT.md Handoff Guide

### What we did
- Added `CONTEXT.md`: a single top-to-bottom resume guide (research summary, current status, file map, run instructions, the open SP-negatives decision, gotchas, paper lineage)
- Recorded the two-checkout layout and golden rule: the WSL checkout (`~/ZSC-Eval`) is authoritative and all git operations happen there; the Windows copy is editing convenience only

---


## 2026-07-11 — Verification of Pushed State + Repo Recovery

### Verification (all passed)
- Functional test: `PartnerEncoder` outputs `[4, 32]`, embeddings L2-normalized to exactly 1.0; all modified modules (`rMAPPOPolicy`, `R_MAPPO`, `SharedReplayBuffer`, `OvercookedRunner`) import cleanly
- InfoNCE at random init printed 2.08 vs the naive log(4)≈1.39 expectation — expected at temperature 0.1 (noisy logits push the untrained loss above log(N)), not a bug
- End-to-end Phase 2 smoke run on `random3` (`--use_partner_encoder --condition_actor_on_partner`): 8,000 timesteps at 269 FPS, exit code 0, zero errors
- CLI coverage check: all 8 encoder args present in `overcooked_config.py`, standard args confirmed in base `config.py`


### Repo recovery & line-ending fix
- Windows checkout was stuck mid-interactive-rebase with 7 unresolved files; a drift audit found zero code lines unique to Windows (WSL was a strict superset), so the rebase was aborted and the checkout hard-reset to the WSL history (fetched via `//wsl.localhost/...` — network fetch times out on Windows)
- Root-caused phantom "modified" files: 23 files had been committed with CRLF before `.gitattributes` existed; fixed with `git add --renormalize .` (commit `988c6ac`, 156 files) and `core.autocrlf=false` on the Windows clone
- Deleted debug-session leftovers (`check_gen.py`, `check_sample.py`, `debug_run.sh`, `remove_debug.py`, stale `scripts/` draft); `test_imports.sh` kept as the tracked smoke test

---


## 2026-07-12 — Method Decision: Staged Encoder Pre-training

### Decision
Resolve the SP-negatives problem (see 06-26 caveat) with **Option 2: pre-train the encoder on frozen FCP population rollouts**, rather than training it jointly during FCP Stage 2.

### Rationale
- **Falsifiable early:** linear-probe accuracy on held-out partners tells us whether the embedding captures partner identity *before* any policy-training compute is spent
- **Data already exists:** the reproduced FCP population checkpoints provide genuinely distinct partners; collection is a standalone script, not training-loop surgery
- **Cleaner ablations and paper story:** mirrors PEARL's staged setup with InfoNCE swapped in for the VAE/ELBO objective
- **Go/no-go gate defined:** probe accuracy well above chance on held-out partners → proceed to policy conditioning; otherwise fix the encoder first

---


## 2026-07-12 — Offline Pre-training Results: Gate PASSED

### Pipeline built and run (all on `random3`)
- `collect_partner_windows.py`: frozen FCP S2 ego paired with each of the 45 frozen
  FCP S1 partner checkpoints (15 runs x init/mid/final), stochastic actions, both seats.
  7,020 windows of 20 steps x 800-dim obs, 156 per partner, balanced.
- `pretrain_partner_encoder.py`: InfoNCE with cross-episode positives (same partner,
  different episode — blocks episode-state shortcuts). Runs sp3/sp8/sp13 (9 checkpoints)
  held out entirely. k-NN probe with episode-split reference/query.

### The bug that mattered
Featurized Overcooked obs are uint8-image-scaled: presence flags = 255, pot timers up
to 5100. Fed raw, they saturate the GRU — first run's loss sat at log(32) and the
held-out probe was 19%. Dividing inputs by 255 fixed it; the scale constant is stored
in the encoder checkpoint so the online path can apply the identical transform.

### Gate results (k-NN probe, held-out partners)
| Metric | 3k steps | 10k steps | chance |
|---|---|---|---|
| held-out partner id | **30.9%** | 28.5% | 11.1% |
| train partner id | 28.5% | 81.4% | 2.8% |
| held-out stage (init/mid/final) | **71.1%** | 66.2% | 33.3% |
| held-out run id | 43.2% | 43.9% | 33.3% |
| seat confound | ~50% | ~50% | 50% |

### Conclusions
- **Gate passed:** ~2.8x chance on unseen-partner identification; seat confound clean.
- **Overfitting past ~3k steps:** train probe climbs to 81% while held-out dips —
  deploy the 3k checkpoint (early stopping).
- **The embedding mostly encodes competence, not identity:** 71% at stage vs 43% at
  run id. For coordination this is arguably the right axis (adapt to partner skill),
  but the write-up should frame it as competence inference.

---

## 2026-07-12 — FCP Stage 2 Conditioning: Smoke Test Passed

### What we did
- Fixed the three crashes hit while first wiring the frozen encoder into Stage 2:
  env reset/step handing back tuples instead of arrays (`np.asarray` guards in
  `trainer_pool.py`), the encoder loaded into a stale policy handle
  (`runner.trainer.trainer_pool[agent].policy`, unwrapping `EvalPolicy`), and a new
  `freeze_partner_encoder` flag so the SP-style joint InfoNCE update is skipped
- Added `shell/train_fcp_s2_contrastive_smoke.sh`: 9,600 steps, 12-partner smoke
  population (`train-smoke.yml`, gitignored under policy_pool), frozen 3k encoder
- Two more launch requirements surfaced: Stage 2 asserts `use_eval` (eval flags are
  mandatory), and `ulimit -n` must be raised or tensorboardX dies with EMFILE at the
  first `log_env` (the stock `train_fcp_stage_2.sh` already does this)

### Result
- Full pipeline ran clean end-to-end: `partner-emb conditioning ACTIVE for
  trainers: ['fcp_adaptive']` logged, frozen encoder loaded (input_scale=255 from
  checkpoint), 2/2 PPO updates at ~93 FPS, eval across all 24 (partner, seat)
  pairings, exit 0
- Shaped reward ~21 after two updates; sparse eval rewards mostly 0 — expected for a
  from-scratch agent on `random3`, this run only validates wiring

---

## 2026-07-12 — Full-Scale Run Prepped: Ready to Launch

### What we did
- Generated the real S2 population yamls (`prep/gen_S2_yml.py random3 fcp` — must be
  run from `zsceval/scripts/`, its `../policy_pool` is CWD-relative): all 15
  `train-s{12,24,36}-sp-{1..5}.yml` now exist locally (gitignored — regenerate on Hyak)
- Added `shell/train_fcp_s2_contrastive.sh`: production launch mirroring the stock
  `train_fcp_stage_2.sh` (same entropy schedule, steps, eval cadence) plus the frozen
  encoder flags; third arg `baseline` drops the encoder for the same-codebase control;
  `SEED_BEGIN`/`SEED_MAX`/`N_THREADS`/`ENCODER_CKPT` env overrides; wandb off; logs
  tee to `~/ZSC-Eval/logs/`
- Launch-validated for 150 s on the real `train-s12-sp-1.yml`: frozen 3k encoder
  loaded, conditioning ACTIVE, 2 PPO updates clean, then killed and the run dir removed
- Updated `CONTEXT.md`: status, run instructions, file map; the SP-negatives
  "open decision" section is now marked RESOLVED

### Compute reality check
- Laptop at 12 threads: ~88 FPS → **~7 days per seed** for the 5e7-step s12 run.
  Hyak remains the sensible venue; a single-seed laptop run is feasible if impatient.

### To launch (in WSL)
```bash
cd ~/ZSC-Eval/zsceval/scripts/overcooked/shell
bash train_fcp_s2_contrastive.sh random3 12            # treatment
bash train_fcp_s2_contrastive.sh random3 12 baseline   # control
```

---

## Next Steps

- [x] Rollout-collection script (done 07-12, `collect_partner_windows.py`)
- [x] Offline InfoNCE pre-training (done 07-12, `pretrain_partner_encoder.py`)
- [x] Go/no-go gate: PASSED — 30.9% held-out partner id vs 11.1% chance (3k ckpt)
- [x] Integrate frozen encoder into FCP Stage 2 training (done 07-12, smoke test
      passed end-to-end; see entry above)
- [x] Prep full-scale launch (done 07-12: population yamls generated,
      `train_fcp_s2_contrastive.sh` launch-validated — see entry above)
- [ ] **RUN IT:** conditioned S2 agent on `random3` (+ `baseline` mode control);
      compare BR-Prox vs FCP baseline 0.635
- [ ] Ablations: zero-embedding control, within-episode adaptation curve
- [ ] Move full training runs and ablations to Hyak (access pending)
