#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zsceval

python - <<'EOF'
from zsceval.algorithms.r_mappo.algorithm.contrastive_encoder import PartnerEncoder, infonce_loss
import torch

# Quick functional test
enc = PartnerEncoder(obs_dim=96, hidden_size=64, emb_dim=32)
dummy = torch.randn(4, 20, 96)   # 4 threads, 20 steps, 96-dim obs
emb = enc(dummy)
print(f"encoder output shape: {emb.shape}")   # expect [4, 32]

anchors = emb
positives = enc(torch.randn(4, 20, 96))
loss = infonce_loss(anchors, positives)
print(f"infonce_loss: {loss.item():.4f}")     # expect ~log(4) ≈ 1.39

from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
print("rMAPPOPolicy import OK")

from zsceval.algorithms.r_mappo.r_mappo import R_MAPPO
print("R_MAPPO import OK")

print("ALL IMPORTS OK")
EOF
