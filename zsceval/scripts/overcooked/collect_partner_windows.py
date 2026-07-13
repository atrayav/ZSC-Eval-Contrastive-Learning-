#!/usr/bin/env python
"""
Collect partner-labeled observation windows for offline contrastive pre-training.

Pairs a frozen ego (an FCP Stage-2 agent) with each frozen FCP Stage-1 partner
checkpoint on one layout, runs stochastic episodes, and saves sliding windows of
the PARTNER's observations labeled by partner identity. The output dataset feeds
offline InfoNCE pre-training of the PartnerEncoder, where positives are windows
from the same partner and negatives are windows from other partners — genuinely
distinct behaviors, unlike self-play rollouts (see RESEARCH_LOG 2026-07-12).

Run in WSL, conda env zsceval, from the repo root:
  python zsceval/scripts/overcooked/collect_partner_windows.py \
      --layout_name random3 --num_agents 2 --seed 1 \
      --episode_length 400 --n_eval_rollout_threads 10 --dummy_batch_size 2 \
      --overcooked_version old --episodes_per_pair 2 \
      --out data/partner_windows/random3.npz
"""
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import zsceval
from zsceval.algorithms.population.utils import EvalPolicy
from zsceval.config import get_config
from zsceval.envs.env_wrappers import ShareSubprocDummyBatchVecEnv
from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.overcooked_config import OLD_LAYOUTS, get_overcooked_args
from zsceval.runner.shared.base_runner import make_trainer_policy_cls
from zsceval.utils.train_util import setup_seed


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            env = Overcooked(all_args, run_dir, rank=rank, evaluation=True)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    return ShareSubprocDummyBatchVecEnv(
        [get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)],
        all_args.dummy_batch_size,
    )


def load_frozen_policy(config_path, actor_path, device):
    """Instantiate a policy from a pickled (args, obs_space, share_obs_space,
    act_space) config and load its frozen actor weights."""
    policy_config = list(pickle.load(open(config_path, "rb")))
    policy_args = policy_config[0]
    _, policy_cls = make_trainer_policy_cls(
        policy_args.algorithm_name,
        use_single_network=policy_args.use_single_network,
    )
    policy = policy_cls(*policy_config, device=device)
    policy.load_checkpoint({"actor": str(actor_path)})
    policy.prep_rollout()
    return EvalPolicy(policy_args, policy)


def parse_args(argv):
    parser = get_config()
    parser = get_overcooked_args(parser)
    parser.add_argument("--episodes_per_pair", type=int, default=2,
                        help="episodes per (partner, seat position) combination")
    parser.add_argument("--window_len", type=int, default=20)
    parser.add_argument("--window_stride", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ego_actor", type=str, default="fcp/s2/fcp-S2-s24/1.pt",
                        help="ego checkpoint, relative to policy_pool/<layout>/")
    parser.add_argument("--ego_config", type=str, default="policy_config/rnn_policy_config.pkl")
    parser.add_argument("--partner_glob", type=str, default="fcp/s1/sp/*_actor.pt")
    parser.add_argument("--partner_config", type=str, default="policy_config/mlp_policy_config.pkl")
    parser.add_argument("--max_rounds", type=int, default=0,
                        help="if >0, stop after this many rounds (smoke testing)")
    all_args = parser.parse_known_args(argv)[0]
    all_args.env_name = "Overcooked"
    # attrs the env reads but that only eval_with_population.py defines as args
    all_args.use_phi = False
    all_args.store_traj = False
    all_args.old_dynamics = all_args.layout_name in OLD_LAYOUTS
    return all_args


def main(argv):
    all_args = parse_args(argv)
    setup_seed(all_args.seed)
    device = torch.device("cpu")
    n_threads = all_args.n_eval_rollout_threads

    pool_root = Path(zsceval.__file__).parent / "policy_pool" / all_args.layout_name
    partner_files = sorted(glob.glob(str(pool_root / all_args.partner_glob)))
    assert partner_files, f"no partner checkpoints match {pool_root / all_args.partner_glob}"
    partner_names = [Path(f).name.replace("_actor.pt", "") for f in partner_files]
    print(f"{len(partner_files)} partners, ego={all_args.ego_actor}", flush=True)

    ego = load_frozen_policy(pool_root / all_args.ego_config, pool_root / all_args.ego_actor, device)
    partners = [
        load_frozen_policy(pool_root / all_args.partner_config, f, device) for f in partner_files
    ]

    run_dir = Path("/tmp/collect_partner_windows")
    run_dir.mkdir(parents=True, exist_ok=True)
    envs = make_eval_env(all_args, run_dir)
    envs.reset_featurize_type([("ppo", "ppo") for _ in range(n_threads)])

    # One combo = (partner index, partner seat). Each combo is one episode per pass.
    combos = [(i, pos) for i in range(len(partners)) for pos in (0, 1)] * all_args.episodes_per_pair
    assert len(combos) % n_threads == 0, (
        f"{len(combos)} episodes must be divisible by {n_threads} threads"
    )
    rounds = [combos[r : r + n_threads] for r in range(0, len(combos), n_threads)]
    if all_args.max_rounds > 0:
        rounds = rounds[: all_args.max_rounds]

    windows, labels, seats = [], [], []
    for r, round_combos in enumerate(rounds):
        involved = {ego} | {partners[i] for i, _ in round_combos}
        for pol in involved:
            pol.reset(n_threads, all_args.num_agents)
        for e, (i, pos) in enumerate(round_combos):
            partners[i].register_control_agent(e, pos)
            ego.register_control_agent(e, 1 - pos)

        obs, _, avail = envs.reset()
        partner_seq = [[] for _ in round_combos]
        for _t in range(all_args.episode_length):
            for e, (i, pos) in enumerate(round_combos):
                partner_seq[e].append(np.asarray(obs[e][pos], dtype=np.float32).reshape(-1))
            actions = np.full((n_threads, all_args.num_agents, 1), fill_value=0).tolist()
            for pol in involved:
                agents = pol.control_agents
                if not agents:
                    continue
                pol.prep_rollout()
                obs_lst = np.stack([obs[e][a] for (e, a) in agents], axis=0)
                avail_lst = np.stack([avail[e][a] for (e, a) in agents], axis=0)
                acts = pol.step(obs_lst, agents, deterministic=False, available_actions=avail_lst)
                for act, (e, a) in zip(acts, agents):
                    actions[e][a] = act
            obs, _, _, _, _infos, avail = envs.step(np.array(actions))

        for e, (i, pos) in enumerate(round_combos):
            seq = np.stack(partner_seq[e])  # (T, obs_dim)
            for s in range(0, seq.shape[0] - all_args.window_len + 1, all_args.window_stride):
                windows.append(seq[s : s + all_args.window_len])
                labels.append(i)
                seats.append(pos)
        print(f"round {r + 1}/{len(rounds)} done, {len(windows)} windows", flush=True)

    envs.close()

    windows = np.stack(windows)
    # Featurized Overcooked obs are small non-negative counts/timers; store
    # compactly when they fit in uint8, otherwise fall back to float16.
    if windows.min() >= 0 and windows.max() < 256 and np.allclose(windows, np.round(windows)):
        windows = windows.astype(np.uint8)
    else:
        windows = windows.astype(np.float16)

    out = Path(all_args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        windows=windows,
        labels=np.array(labels, dtype=np.int64),
        seats=np.array(seats, dtype=np.int64),
        partner_names=np.array(partner_names),
    )
    print(
        f"saved {windows.shape} ({windows.dtype}) windows, "
        f"{len(set(labels))} partners -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
