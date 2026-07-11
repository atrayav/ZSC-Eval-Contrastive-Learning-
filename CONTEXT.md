# Project Context — ZSC-Eval Contrastive Partner Encoder

> **Purpose of this file:** a single place to get anyone (you, a collaborator, or an
> AI assistant in a fresh session) up to speed fast. Read this top-to-bottom and you
> should be able to continue the work without re-deriving anything.
>
> Last updated: 2026-07-10.

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

- Encoder stack is **fully implemented, commented, and pushed.** Dry-run passes on `random3`.
- **Not yet run at full scale.** No results for the contrastive method yet.
- Baselines already reproduced (see `RESEARCH_LOG.md`): e.g. `random3` SP=0.10, FCP=0.64, MEP=0.70.

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
| `zsceval/scripts/overcooked/shell/train_contrastive_sp.sh` | **NEW.** Launch script. |

## 5. How to run it

```bash
# in WSL:
source ~/miniconda3/etc/profile.d/conda.sh && conda activate zsceval
cd ~/ZSC-Eval
# Phase 1 — train the encoder only (actor ignores embedding):
bash zsceval/scripts/overcooked/shell/train_contrastive_sp.sh random3
# Phase 2 — also condition the actor on the embedding:
bash zsceval/scripts/overcooked/shell/train_contrastive_sp.sh random3 --condition
```

**Two-switch design:** `--use_partner_encoder` alone = Phase 1 (learn hunches).
Add `--condition_actor_on_partner` = Phase 2 (act on hunches). Other args:
`--partner_emb_dim 32 --encoder_context_len 20 --encoder_hidden_size 64
--infonce_temperature 0.1 --encoder_lr 1e-3 --infonce_coef 1.0`.

## 6. ⚠️ THE open decision — start here

The encoder trains on **self-play** rollouts, where every thread runs the *same* policy.
So InfoNCE's "negatives" are not distinct partners — the encoder can only learn to tell
trajectory/state noise apart, **not partner identity.** This makes the core claim
untestable as currently wired.

**Fix:** move encoder training from SP → **FCP Stage 2**, where each thread is paired with
a genuinely distinct population partner. Two options, not yet chosen:
1. Train the encoder jointly during FCP S2 (more integrated).
2. Pre-train the encoder on frozen FCP population rollouts, then condition the policy
   separately (cleaner to debug; closer to PEARL's staged setup).

Everything downstream (which files change, whether `train_contrastive_sp.sh` forks into a
`_fcp` variant) depends on this choice.

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
