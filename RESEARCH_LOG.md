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


## Next Steps

- [ ] **Method design (priority):** decide positive/negative construction for partner embeddings — FCP population gives distinct partner policies per thread, making InfoNCE meaningful
- [ ] Adjust encoder training to use FCP-style population rather than SP
- [ ] Run small local smoke test with new pair construction
- [ ] Move full training runs and ablations to Hyak
- [ ] Get Hyak/UWRCC cluster access (requested, pending approval)
