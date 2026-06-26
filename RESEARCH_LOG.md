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

## Next Steps

- [ ] Share results with PI, align on primary layout and training approach
- [ ] Get Hyak/UWRCC cluster access (requested, pending approval)
- [ ] Read FCP paper (arxiv 2110.08176) and relevant partner-modeling literature
- [ ] Implement contrastive partner encoder (`contrastive_encoder.py`)
  - InfoNCE loss: anchor/positive/negative trajectory clips
  - Feed partner embedding into ego policy actor
  - Ablation: constant (zeroed) embedding as control
- [ ] Train on Hyak with GPU, evaluate against SP/FCP/MEP baselines
