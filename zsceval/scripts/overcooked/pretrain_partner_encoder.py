#!/usr/bin/env python
"""
Offline InfoNCE pre-training of the PartnerEncoder on partner-labeled windows.

Consumes the dataset from collect_partner_windows.py. Positives are two windows
of the SAME partner drawn from DIFFERENT episodes (so the encoder cannot cheat
via episode-specific state); negatives are the other partners in the batch.

Entire runs (all init/mid/final checkpoints of a self-play seed) are held out
from training and used for the go/no-go probe: k-NN partner classification on
held-out partners, split by episode (reference episodes vs query episodes).

Run in WSL, conda env zsceval, from the repo root:
  python zsceval/scripts/overcooked/pretrain_partner_encoder.py \
      --data data/partner_windows/random3.npz \
      --out data/partner_windows/encoder_random3.pt
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from zsceval.algorithms.r_mappo.algorithm.contrastive_encoder import (
    PartnerEncoder,
    infonce_loss,
)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--held_out_runs", type=str, default="sp3,sp8,sp13",
                   help="comma-separated self-play run prefixes excluded from training")
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_partners", type=int, default=32)
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--input_scale", type=float, default=255.0,
                   help="divide observations by this; the featurized obs are "
                        "uint8-image-scaled (presence flag = 255, timers = 255*t)")
    return p.parse_args(argv)


def episode_ids(labels, windows_per_episode):
    """Windows are written per-episode in contiguous blocks by the collector."""
    ep = np.arange(len(labels)) // windows_per_episode
    # sanity: label constant within each block
    for e in range(ep.max() + 1):
        blk = labels[ep == e]
        assert (blk == blk[0]).all(), f"episode block {e} mixes partners"
    return ep


def embed_all(encoder, windows, input_scale, batch=256):
    encoder.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(windows), batch):
            x = torch.as_tensor(windows[s : s + batch], dtype=torch.float32) / input_scale
            out.append(encoder(x))
    return torch.cat(out).numpy()


def knn_accuracy(ref_emb, ref_labels, query_emb, query_labels, k):
    """Cosine k-NN majority vote (embeddings are already L2-normalized)."""
    sims = query_emb @ ref_emb.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    correct = 0
    for i, idxs in enumerate(topk):
        votes = ref_labels[idxs]
        pred = np.bincount(votes).argmax()
        correct += pred == query_labels[i]
    return correct / len(query_labels)


def episode_split_probe(emb, labels, ep, classes, k):
    """Per class: first half of its episodes are k-NN references, the rest are
    queries. Returns accuracy over all queries of the given classes."""
    ref_idx, query_idx = [], []
    for c in classes:
        eps = sorted(set(ep[labels == c]))
        ref_eps = set(eps[: len(eps) // 2])
        for e in eps:
            idx = np.where((labels == c) & (ep == e))[0]
            (ref_idx if e in ref_eps else query_idx).append(idx)
    ref_idx, query_idx = np.concatenate(ref_idx), np.concatenate(query_idx)
    return knn_accuracy(emb[ref_idx], labels[ref_idx], emb[query_idx], labels[query_idx], k)


def main(argv):
    args = parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    windows, labels, seats = d["windows"], d["labels"], d["seats"]
    names = [str(n) for n in d["partner_names"]]
    n_classes = len(names)
    obs_dim = windows.shape[2]

    # contiguous per-episode blocks -> infer block size from stride pattern
    windows_per_episode = np.flatnonzero(np.diff(labels) != 0)
    windows_per_episode = int(windows_per_episode[0] + 1) if len(windows_per_episode) else len(labels)
    ep = episode_ids(labels, windows_per_episode)

    held_runs = set(args.held_out_runs.split(","))
    held_classes = [i for i, n in enumerate(names) if n.split("_")[0] in held_runs]
    train_classes = [i for i in range(n_classes) if i not in held_classes]
    print(f"{n_classes} partners total, train on {len(train_classes)}, "
          f"hold out {len(held_classes)} ({sorted(names[i] for i in held_classes)})", flush=True)

    # index windows by (class, episode) for cross-episode positive sampling
    by_class_ep = defaultdict(lambda: defaultdict(list))
    for i, (c, e) in enumerate(zip(labels, ep)):
        by_class_ep[c][e].append(i)
    by_class_ep = {c: {e: np.array(v) for e, v in eps.items()} for c, eps in by_class_ep.items()}

    encoder = PartnerEncoder(obs_dim=obs_dim, hidden_size=args.hidden_size, emb_dim=args.emb_dim)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    encoder.train()
    loss_log = []
    for step in range(1, args.steps + 1):
        cs = np.random.choice(train_classes, size=args.batch_partners, replace=False)
        anchor_idx, pos_idx = [], []
        for c in cs:
            eps = list(by_class_ep[c].keys())
            e1, e2 = np.random.choice(eps, size=2, replace=False)
            anchor_idx.append(np.random.choice(by_class_ep[c][e1]))
            pos_idx.append(np.random.choice(by_class_ep[c][e2]))
        xa = torch.as_tensor(windows[anchor_idx], dtype=torch.float32) / args.input_scale
        xp = torch.as_tensor(windows[pos_idx], dtype=torch.float32) / args.input_scale
        loss = infonce_loss(encoder(xa), encoder(xp), temperature=args.temperature)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_log.append(loss.item())
        if step % 200 == 0:
            print(f"step {step}/{args.steps} loss {np.mean(loss_log[-200:]):.4f}", flush=True)

    emb = embed_all(encoder, windows, args.input_scale)

    # coarser labelings for diagnostics on the held-out windows:
    # stage (init/mid/final, chance 1/3) and run identity (chance 1/n_held_runs)
    stage_names = ["init", "mid", "final"]
    stage_of = np.array([stage_names.index(n.split("_")[1]) for n in names])
    run_names = sorted({n.split("_")[0] for n in names})
    run_of = np.array([run_names.index(n.split("_")[0]) for n in names])
    held_mask = np.isin(labels, held_classes)
    held_stage = stage_of[labels]
    held_run = run_of[labels]

    def held_probe(coarse_labels, classes):
        masked = np.where(held_mask, coarse_labels, -1)
        return episode_split_probe(emb, masked, ep, classes, args.knn_k)

    metrics = {
        "loss_first_200": float(np.mean(loss_log[:200])),
        "loss_last_200": float(np.mean(loss_log[-200:])),
        "held_out_probe_acc": episode_split_probe(emb, labels, ep, held_classes, args.knn_k),
        "held_out_chance": 1.0 / len(held_classes),
        "train_probe_acc": episode_split_probe(emb, labels, ep, train_classes, args.knn_k),
        "train_chance": 1.0 / len(train_classes),
        # confound check: can the embedding predict seat position instead of identity?
        "seat_probe_acc": episode_split_probe(emb, seats, ep, [0, 1], args.knn_k),
        # what does the embedding capture on held-out runs: training stage vs run identity?
        "held_out_stage_probe_acc": held_probe(held_stage, [0, 1, 2]),
        "held_out_run_probe_acc": held_probe(
            held_run, sorted({run_of[c] for c in held_classes})
        ),
        "held_out_runs": sorted(held_runs),
        "input_scale": args.input_scale,
    }
    print(json.dumps(metrics, indent=2), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "input_scale": args.input_scale,
            "obs_dim": obs_dim,
            "hidden_size": args.hidden_size,
            "emb_dim": args.emb_dim,
        },
        out,
    )
    with open(out.with_suffix(".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved encoder -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
