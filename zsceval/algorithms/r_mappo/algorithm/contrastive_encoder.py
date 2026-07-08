"""
Contrastive partner encoder for zero-shot coordination.

This module is the core of the research extension. It learns to look at a
short window of a partner's recent observations and summarize "what kind of
partner is this" into a small fixed-size vector (the partner embedding).

Two pieces live here:
  - PartnerEncoder: the network that turns partner observations into an embedding.
  - infonce_loss:  the contrastive objective that teaches the encoder to make
                   same-partner embeddings similar and different-partner
                   embeddings dissimilar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartnerEncoder(nn.Module):
    """
    GRU-based encoder: maps a window of partner observations to a fixed-size embedding.

    Accepts a sequence of T partner obs steps and returns a single normalized vector
    that summarizes partner behavior. Used for contrastive (InfoNCE) training and
    optionally for conditioning the ego actor.
    """

    def __init__(self, obs_dim: int, hidden_size: int, emb_dim: int):
        super().__init__()
        # The GRU reads the partner's observations one timestep at a time,
        # keeping a running "memory" of what it has seen so far. batch_first
        # means we feed tensors shaped [batch, time, obs_dim].
        self.gru = nn.GRU(obs_dim, hidden_size, num_layers=1, batch_first=True)
        # A small MLP head that compresses the GRU's final memory down to the
        # embedding size. The ReLU in the middle lets it learn non-linear
        # summaries rather than a plain linear projection.
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

    def forward(self, obs_window: torch.Tensor) -> torch.Tensor:
        """
        obs_window: [batch, T, obs_dim]
        Returns [batch, emb_dim] L2-normalized embeddings.
        """
        _, h_n = self.gru(obs_window)    # h_n: [1, batch, hidden_size]
        emb = self.proj(h_n.squeeze(0))  # [batch, emb_dim]
        return F.normalize(emb, dim=-1)


def infonce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss.

    anchors, positives: [N, emb_dim], L2-normalized.
    Row i is a matched anchor/positive pair; all other rows are negatives.
    Requires N >= 2 (returns zero otherwise).
    """
    N = anchors.shape[0]
    if N < 2:
        return anchors.new_zeros(()).squeeze()

    sim = torch.mm(anchors, positives.T) / temperature  # [N, N]
    labels = torch.arange(N, device=anchors.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
    return loss
