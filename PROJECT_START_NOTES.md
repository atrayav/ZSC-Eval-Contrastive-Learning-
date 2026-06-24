# Multi-Agent RL Project Start Notes

## Research Goal

Test whether contrastively learned partner representations improve zero-shot coordination over population-only baselines in Overcooked-style multi-agent RL.

Core claim to validate:

> A policy conditioned on an inferred partner embedding should coordinate better with held-out partners than a single population-trained generalist policy.

## Repo

Official toolkit:

- `sjtu-marl/ZSC-Eval`
- Local clone: `C:\Users\varun\ZSC-Eval`

The repo includes:

- Overcooked and multi-recipe Overcooked environments
- Google Research Football support
- Baseline implementations: `FCP`, `MEP`, `TrajeDi`, `HSP`, `COLE`, `E3T`, `SP`
- Evaluation machinery for ZSC, including behavior-preferring partners and BR-Prox-style evaluation
- Pretrained policy pool is expected at `zsceval/policy_pool`

## Important Setup Constraint

This repo expects a Linux-like workflow:

- `conda`
- `bash`
- PyTorch/CUDA
- shell scripts under `zsceval/scripts/overcooked/shell`

Native Windows PowerShell is not the ideal runtime. Use one of:

- WSL2 Ubuntu on this laptop for local debugging
- Hyak/Tillicum for GPU training and larger sweeps

Current local status:

- Repo cloned successfully.
- `conda` is not currently available in this PowerShell environment.

## Key Files

- `README.md`: official setup and experiment instructions
- `environment.yml`: conda environment definition, Python 3.9, PyTorch CUDA 11.8
- `requirements.txt`: Python package dependencies
- `zsceval/scripts/overcooked/shell/`: bash entrypoints for training/eval
- `zsceval/scripts/overcooked/eval/`: evaluation scripts
- `zsceval/utils/bias_agent_vars.py`: selected biased partner IDs per layout
- `zsceval/overcooked_config.py`: supported old-layout names

## Layouts To Know

Old/single-recipe layouts:

- `random0`
- `random0_medium`
- `random1`
- `random3`
- `small_corridor`
- `unident_s`

Multi-recipe layouts:

- `random0_m`
- `random1_m`
- `random3_m`

Start with an old/small layout for debugging, likely `random0` or `unident_s`. Use multi-recipe layouts later because they are more relevant to avoiding saturation.

## First Milestone

Do not implement the contrastive agent yet.

First milestone:

> Run ZSC-Eval Overcooked evaluation for an existing baseline and produce one score table by layout and algorithm.

Minimum path:

1. Install environment.
2. Download pretrained policy pool.
3. Run an existing baseline evaluation, probably `FCP` on `random0` or `unident_s`.
4. Extract results into a small table.
5. Confirm that scores look sensible.

## Official Install Commands

From repo root:

```bash
conda env create -f environment.yml
conda activate zsceval
```

Download pretrained policy pool:

```bash
cd zsceval
git clone https://huggingface.co/Leoxxxxh/ZSC-Eval-policy_pool policy_pool
```

Run Overcooked eval commands from:

```bash
cd zsceval/scripts/overcooked
```

Official example:

```bash
bash shell/eval_with_bias_agents.sh {layout} fcp
cd ..
python eval/extract_results.py -a fcp -l {layout}
```

## Research Implementation Sequence

1. Reproduce baseline evaluation.
2. Reproduce or reuse FCP/MEP pretrained models.
3. Characterize baseline weaknesses by layout and partner type.
4. Implement low-risk partner-discrimination InfoNCE:
   - anchor: clip of partner behavior
   - positive: another clip from same partner
   - negatives: clips from different partners
5. Feed partner embedding into the ego policy.
6. Add ablation: same architecture, constant partner embedding.
7. Only after this works, try value-grounded / goal-conditioned contrastive RL.

## Key Failure Mode

The contrastive partner encoder may memorize training-partner identity instead of learning transferable behavioral style.

Required checks:

- Probe accuracy on held-out partners, not only training partners.
- ZSC score on held-out partner population.
- Constant-embedding ablation.
- Within-episode adaptation curve.

