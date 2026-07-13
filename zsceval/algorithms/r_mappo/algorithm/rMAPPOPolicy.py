import numpy as np
import torch
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.contrastive_encoder import PartnerEncoder
from zsceval.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from zsceval.utils.util import get_shape_from_obs_space, update_linear_schedule


class ExDataParallel(torch.nn.DataParallel):
    """DataParallel that forwards unknown attributes to the wrapped module.

    Plain torch.nn.DataParallel only exposes its own attributes, which breaks
    code that reaches into custom submodules (e.g. ``actor.base.cnn``). This
    subclass lets callers stay oblivious to whether multi-GPU forwarding is on.
    """

    def __getattr__(self, name):
        try:
            # First try DataParallel's own attributes (module, device_ids, ...).
            return super().__getattr__(name)
        except AttributeError:
            # Not found -> fall through to the wrapped module itself.
            return getattr(self.module, name)


class R_MAPPOPolicy:
    """MAPPO policy: a decentralized actor plus a centralized critic.

    The actor maps each agent's LOCAL observation to an action distribution;
    the critic maps the SHARED (centralized) observation to a value estimate.
    This asymmetry is the "centralized training, decentralized execution"
    (CTDE) core of MAPPO. Training code talks to this object rather than to
    the networks directly.
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        # Where the networks live (cpu/cuda). Inputs are moved here on use.
        self.device = device
        # Separate learning rates: the value loss and the clipped surrogate
        # loss have different scales, so actor and critic each get their own
        # optimizer and step size below.
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        # Adam stability knobs shared by both optimizers.
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        # Gym spaces are kept so networks can be rebuilt or checkpointed and
        # so wrappers (e.g. the population pool) can introspect this policy.
        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.data_parallel = getattr(args, "data_parallel", False)

        # Actor consumes local obs only; critic consumes centralized
        # share_obs. The critic is discarded at execution time, so a richer
        # critic input never leaks privileged info into acting.
        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        # Two independent Adam optimizers: PPO alternates the clipped
        # surrogate update (actor) and the value regression update (critic),
        # and nothing forces them to share a schedule.
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        # Partner encoder (optional contrastive extension).
        # The Policy object owns BOTH the actor/critic and this new encoder, so
        # it is the natural place to build it and hand it its own optimizer.
        self.use_partner_encoder = getattr(args, "use_partner_encoder", False)
        if self.use_partner_encoder:
            obs_shape = get_shape_from_obs_space(obs_space)
            # Overcooked observations are a 3-D grid (e.g. 8x5x20). The GRU
            # wants a flat vector per timestep, so collapse all obs dims into
            # one number (np.prod) - this also handles already-1-D obs.
            obs_dim = int(np.prod(obs_shape))  # flatten (H, W, C) or 1-D obs
            self.encoder = PartnerEncoder(
                obs_dim=obs_dim,
                hidden_size=getattr(args, "encoder_hidden_size", 64),
                emb_dim=getattr(args, "partner_emb_dim", 32),
            ).to(device)
            # Separate optimizer: the encoder learns from the contrastive loss
            # on its own schedule, independent of the PPO actor/critic updates.
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(),
                lr=getattr(args, "encoder_lr", 1e-3),
            )
        else:
            # Feature off: keep the attributes so the rest of the code can
            # check "is self.encoder None?" without special-casing.
            self.encoder = None
            self.encoder_optimizer = None

    def to_parallel(self):
        # Fan forward passes out across all visible GPUs. Each top-level
        # child of actor/critic is wrapped individually (rather than the
        # whole network) so the module attribute layout stays unchanged for
        # code that reaches inside, e.g. actor.base or actor.act.
        if self.data_parallel:
            logger.warning(
                f"Use Data Parallel for Forwarding in devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
            )
            for name, children in self.actor.named_children():
                setattr(self.actor, name, ExDataParallel(children))
            for name, children in self.critic.named_children():
                setattr(self.critic, name, ExDataParallel(children))

    def lr_decay(self, episode, episodes):
        # Linearly anneal both learning rates from their initial value toward
        # zero as training progresses (episode / episodes elapsed fraction).
        # Called once per training episode by the runner.
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        # partner_emb: the current partner hunch, forwarded straight to the
        # actor so acting can be partner-aware. None => baseline behavior.
        partner_emb=None,
        **kwargs,
    ):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic, partner_emb=partner_emb
        )
        values, rnn_states_critic = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks, task_id=None):
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values

    def evaluate_actions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
        partner_emb=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks, partner_emb=partner_emb)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
            rnn_states_actor,
        ) = self.actor.evaluate_transitions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, rnn_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, rnn_states_actor

    def act(
        self,
        obs,
        rnn_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        partner_emb=None,
        **kwargs,
    ):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, partner_emb=partner_emb)
        return actions, rnn_states_actor

    def get_probs(self, obs, rnn_states_actor, masks, available_actions=None):
        action_probs, rnn_states_actor = self.actor.get_probs(
            obs, rnn_states_actor, masks, available_actions=available_actions
        )
        return action_probs, rnn_states_actor

    def get_action_log_probs(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action_log_probs, _, _, rnn_states_actor = self.actor.get_action_log_probs(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, rnn_states_actor

    def load_checkpoint(self, ckpt_path):
        if "actor" in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if "critic" in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        if self.encoder is not None:
            self.encoder.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
        # Put the encoder in eval mode too, so it produces hunches during data
        # collection without dropout/batchnorm training behavior kicking in.
        if self.encoder is not None:
            self.encoder.eval()
