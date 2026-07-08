import argparse

from zsceval.config import scientific_notation

OLD_LAYOUTS = [
    "random0",
    "random0_medium",
    "random1",
    "random3",
    "small_corridor",
    "unident_s",
]


def get_overcooked_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--layout_name",
        type=str,
        default="cramped_room",
        help="Name of Submap, 40+ in choice. See /src/data/layouts/.",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument(
        "--use_timestep_feature",
        action="store_true",
        default=False,
        help="add timestep as a feature",
    )
    parser.add_argument(
        "--use_identity_feature",
        action="store_true",
        default=False,
        help="add id as a feature",
    )
    parser.add_argument(
        "--use_agent_policy_id",
        default=False,
        action="store_true",
        help="Add policy id into share obs, default False",
    )
    parser.add_argument(
        "--initial_reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_factor",
        type=float,
        default=1.0,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--reward_shaping_horizon",
        type=scientific_notation,
        default=2.5e6,
        help="Shaping factor of potential dense reward.",
    )
    parser.add_argument(
        "--random_start_prob",
        default=0.0,
        type=float,
        help="Probability to use a random start state, default 0.",
    )
    parser.add_argument("--use_random_terrain_state", default=False, action="store_true")
    parser.add_argument("--use_random_player_pos", default=False, action="store_true")
    parser.add_argument("--overcooked_version", default="old", type=str, choices=["new", "old"])
    parser.add_argument("--random_index", default=False, action="store_true")
    parser.add_argument("--use_hsp", default=False, action="store_true")
    parser.add_argument("--w0_offset", default=0, type=int)
    parser.add_argument(
        "--w0",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 0 in overcooked env.",
    )
    parser.add_argument(
        "--w1",
        type=str,
        default="1,1,1,1",
        help="Weight vector of dense reward 1 in overcooked env.",
    )

    parser.add_argument("--num_initial_state", type=int, default=5)
    parser.add_argument("--replay_return_threshold", type=float, default=0.75)

    # --- Contrastive partner encoder (Varun Atraya extension) ---
    # Two independent switches define the training phase:
    #   Phase 1: --use_partner_encoder alone. The encoder trains via InfoNCE,
    #            but the actor ignores the embedding (learns to form hunches).
    #   Phase 2: add --condition_actor_on_partner. The embedding is fed into
    #            the actor so behavior can depend on the inferred partner.
    # The remaining args (dims, temperature, lr, coef) are hyperparameters.
    parser.add_argument(
        "--use_partner_encoder",
        action="store_true",
        default=False,
        help="Enable contrastive partner encoder (InfoNCE training).",
    )
    parser.add_argument(
        "--partner_emb_dim",
        type=int,
        default=32,
        help="Dimension of the partner embedding vector.",
    )
    parser.add_argument(
        "--encoder_context_len",
        type=int,
        default=20,
        help="Number of recent partner obs steps to feed into the encoder.",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=64,
        help="GRU hidden size inside the partner encoder.",
    )
    parser.add_argument(
        "--infonce_temperature",
        type=float,
        default=0.1,
        help="Temperature for InfoNCE contrastive loss.",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the partner encoder optimizer.",
    )
    parser.add_argument(
        "--infonce_coef",
        type=float,
        default=1.0,
        help="Weight of the InfoNCE loss relative to PPO loss.",
    )
    parser.add_argument(
        "--condition_actor_on_partner",
        action="store_true",
        default=False,
        help="Concatenate partner embedding into actor features (requires use_partner_encoder).",
    )

    return parser
