from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.r_mappo.algorithm.contrastive_encoder import infonce_loss
from zsceval.algorithms.utils.util import check
from zsceval.utils.util import get_gard_norm, huber_loss, mse_loss
from zsceval.utils.valuenorm import ValueNorm

from .algorithm.rMAPPOPolicy import R_MAPPOPolicy


class R_MAPPO:
    def __init__(self, args, policy: R_MAPPOPolicy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.policy_value_loss_coef = args.policy_value_loss_coef
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coefs = args.entropy_coefs
        self.entropy_coef_horizons = args.entropy_coef_horizons

        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.share_policy = args.share_policy

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_policy_vhead = args.use_policy_vhead
        self._use_task_v_out = getattr(args, "use_task_v_out", False)

        self.use_partner_encoder = getattr(args, "use_partner_encoder", False)
        self.infonce_temperature = getattr(args, "infonce_temperature", 0.1)
        self.infonce_coef = getattr(args, "infonce_coef", 1.0)

        assert (
            self._use_popart and self._use_valuenorm
        ) == False, "self._use_popart and self._use_valuenorm can not be set True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
            if self._use_policy_vhead:
                self.policy_value_normalizer = self.policy.actor.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
            if self._use_policy_vhead:
                self.policy_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            if self._use_policy_vhead:
                self.policy_value_normalizer = None

    def adapt_entropy_coef(self, num_steps: int):
        n = len(self.entropy_coef_horizons)
        for i in range(n - 1):
            if self.entropy_coef_horizons[i] <= num_steps < self.entropy_coef_horizons[i + 1]:
                start_steps = self.entropy_coef_horizons[i]
                end_steps = self.entropy_coef_horizons[i + 1]
                start_coef = self.entropy_coefs[i]
                end_coef = self.entropy_coefs[i + 1]
                fraction = (num_steps - start_steps) / (end_steps - start_steps)
                self.entropy_coef = (1 - fraction) * start_coef + fraction * end_coef
                break
        else:  # num_steps >= the last horizon
            self.entropy_coef = self.entropy_coefs[-1]

        logger.trace(
            f"entropy_coef: {self.entropy_coef:.5f} at {num_steps} steps, schedule {self.entropy_coefs} {self.entropy_coef_horizons}"
        )

    def cal_value_loss(
        self,
        value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)

        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(
        self,
        sample,
        turn_on: bool = True,
        actor_zero_grad: bool = True,
        critic_zero_grad: bool = True,
    ):
        # Backward-compatibility seatbelt: older buffer generators yielded 13
        # fields (no partner embedding). If we see a 13-field sample, append a
        # None so the unpacking below still works and conditioning is simply
        # treated as "no embedding" for that batch.
        if len(sample) == 13:
            sample = (*sample, None)

        if self.share_policy:
            (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                other_policy_id_batch,
                partner_embs_batch,
            ) = sample
        else:
            (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
                _,
                partner_embs_batch,
            ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        if self._use_task_v_out and other_policy_id_batch is not None:
            other_policy_id_batch = check(other_policy_id_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        (
            values,
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            task_id=other_policy_id_batch if self._use_task_v_out else None,
            # Feed back the exact embedding stored at collection time so the
            # recomputed log-probs match the ones used to pick these actions.
            partner_emb=partner_embs_batch,
        )
        # actor update
        # ratio = new_prob / old_prob. This is only correct when both probs
        # were computed with identical inputs - hence passing partner_emb above.
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            policy_value_loss = self.cal_value_loss(
                self.policy_value_normalizer,
                policy_values,
                value_preds_batch,
                return_batch,
                active_masks_batch,
            )
            policy_loss = policy_action_loss + policy_value_loss * self.policy_value_loss_coef
        else:
            policy_loss = policy_action_loss

        if actor_zero_grad:
            # logger.debug("actor zero grad")
            self.policy.actor_optimizer.zero_grad()

        if turn_on:
            loss = policy_loss - dist_entropy * self.entropy_coef
            loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()
        upper_rate = torch.sum((1.0 * ratio) > (1 + self.clip_param)) / torch.numel(ratio)
        lower_rate = torch.sum((1.0 * ratio) < (1 - self.clip_param)) / torch.numel(ratio)

        upper_rate = torch.sum((1.0 * ratio) > (1 + self.clip_param)) / torch.numel(ratio)
        lower_rate = torch.sum((1.0 * ratio) < (1 - self.clip_param)) / torch.numel(ratio)

        # critic update
        value_loss = self.cal_value_loss(
            self.value_normalizer,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch,
        )

        if critic_zero_grad:
            self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
            upper_rate,
            lower_rate,
            self.entropy_coef,
        )

    def update_actor(self):
        if self._use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

    def compute_advantages(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        return advantages

    def update_encoder(self, buffer) -> float:
        """
        Train the partner encoder with InfoNCE loss for one gradient step.

        Uses the full episode buffer: the first half of each thread's partner
        trajectory is the anchor, the second half is the positive.
        All N threads serve as negatives for each other.

        buffer.obs shape: [T+1, N, M, obs_dim]
        Partner obs for ego (agent index 0) = buffer.obs[:, :, 1, :]
        """
        if self.policy.encoder is None:
            return 0.0

        obs_full = buffer.obs[:-1]           # [T, N, M, ...obs_shape...]
        T, N, M = obs_full.shape[:3]

        if N < 2 or T < 2:
            return 0.0

        # Flatten per-obs dimensions so the GRU sees [batch, T, flat_obs_dim].
        # agent_idx 0 is ego; agent_idx 1 is partner (for ego at position 0).
        obs_flat = obs_full.reshape(T, N, M, -1)   # [T, N, M, flat_obs_dim]
        partner_obs = torch.FloatTensor(obs_flat[:, :, 1, :]).to(self.device)
        # partner_obs: [T, N, flat_obs_dim]
        partner_obs = partner_obs.permute(1, 0, 2)  # [N, T, flat_obs_dim]

        half = T // 2
        self.policy.encoder.train()
        anchor_emb = self.policy.encoder(partner_obs[:, :half, :])    # [N, emb_dim]
        pos_emb = self.policy.encoder(partner_obs[:, half:, :])       # [N, emb_dim]

        loss = infonce_loss(anchor_emb, pos_emb, temperature=self.infonce_temperature)
        loss = loss * self.infonce_coef

        self.policy.encoder_optimizer.zero_grad()
        loss.backward()
        if self._use_max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.encoder.parameters(), self.max_grad_norm)
        self.policy.encoder_optimizer.step()

        return loss.item()

    def train(self, buffer, turn_on=True, **kwargs):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = defaultdict(float)

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["upper_clip_rate"] = 0
        train_info["lower_clip_rate"] = 0
        train_info["entropy_coef"] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                    upper_rate,
                    lower_rate,
                    entropy_coef,
                ) = self.ppo_update(sample, turn_on, **kwargs)

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()

                if int(torch.__version__[2]) < 5:
                    train_info["actor_grad_norm"] += actor_grad_norm.item()
                    train_info["critic_grad_norm"] += critic_grad_norm.item()
                else:
                    train_info["actor_grad_norm"] += actor_grad_norm.item()
                    train_info["critic_grad_norm"] += critic_grad_norm.item()

                train_info["ratio"] += ratio.mean().item()
                train_info["upper_clip_rate"] += upper_rate.item()
                train_info["lower_clip_rate"] += lower_rate.item()
                train_info["entropy_coef"] += entropy_coef

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        if self.use_partner_encoder:
            train_info["encoder_loss"] = self.update_encoder(buffer)

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        if self.policy.encoder is not None:
            self.policy.encoder.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        if self.policy.encoder is not None:
            self.policy.encoder.eval()

    def to(self, device):
        self.policy.to(device)
