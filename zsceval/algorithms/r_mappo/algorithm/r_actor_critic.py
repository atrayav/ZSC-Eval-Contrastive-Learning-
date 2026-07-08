import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from zsceval.algorithms.utils.act import ACTLayer
from zsceval.algorithms.utils.cnn import CNNBase
from zsceval.algorithms.utils.mix import MIXBase
from zsceval.algorithms.utils.mlp import MLPBase, MLPLayer
from zsceval.algorithms.utils.popart import PopArt
from zsceval.algorithms.utils.rnn import RNNLayer
from zsceval.algorithms.utils.util import check, init
from zsceval.utils.util import get_shape_from_obs_space


class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._influence_layer_N = args.influence_layer_N
        self._use_policy_vhead = args.use_policy_vhead
        self._use_popart = args.use_popart
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)

        # logger.debug(f"actor obs shape: {obs_shape}")
        logger.trace(f"actor obs shape: {obs_shape}")
        if "Dict" in obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.base = (
                CNNBase(args, obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(obs_shape) == 3
                else MLPBase(
                    args,
                    obs_shape,
                    use_attn_internal=args.use_attn_internal,
                    use_cat_self=True,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        # Partner embedding injected right before the action head.
        # When condition_actor_on_partner is True, the actor receives an extra
        # conditioning vector of size partner_emb_dim concatenated to actor_features.
        self.partner_emb_dim = getattr(args, "partner_emb_dim", 0)
        self.condition_actor = getattr(args, "condition_actor_on_partner", False)
        act_input_size = input_size + (self.partner_emb_dim if self.condition_actor else 0)

        self.act = ACTLayer(action_space, act_input_size, self._use_orthogonal, self._gain)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_policy_vhead:
            if self._use_popart:
                self.v_out = init_(PopArt(act_input_size, 1, device=device))
            else:
                self.v_out = init_(nn.Linear(act_input_size, 1))

        self.to(device)

    def _cat_partner_emb(self, actor_features: torch.Tensor, partner_emb) -> torch.Tensor:
        """Concatenate partner embedding if actor conditioning is enabled.

        This is the single point where the partner "hunch" enters the actor.
        It is called in forward() and both evaluate_* paths, so all three must
        behave identically - otherwise the log-probs computed at action time
        and at training time would disagree and break PPO.
        """
        # Feature disabled: return the actor features untouched so behaviour
        # exactly matches the unmodified baseline network.
        if not self.condition_actor or self.partner_emb_dim == 0:
            return actor_features
        # Conditioning is on but no embedding was supplied (e.g. Phase 1, or a
        # warmup step before the window is full): pad with zeros so the input
        # width the action head expects stays constant and nothing crashes.
        if partner_emb is None:
            pad = torch.zeros(actor_features.shape[0], self.partner_emb_dim, **self.tpdv)
        else:
            pad = check(partner_emb).to(**self.tpdv)
        # Glue the embedding onto the end of the per-observation features.
        return torch.cat([actor_features, pad], dim=-1)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, partner_emb=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        # Inject the partner hunch here, after the RNN/MLP have summarized the
        # observation but before the action head - so the choice of action can
        # depend on "who am I working with", not just "what does the kitchen
        # look like". This is the live acting path used during rollouts.
        actor_features = self._cat_partner_emb(actor_features, partner_emb)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_transitions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, partner_emb=None):
        # ! only work for rnn model
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)

        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        # Training path: re-evaluate stored actions. The embedding here must be
        # the SAME one used when the action was taken (passed down from the
        # buffer), or PPO's importance ratio is computed on mismatched inputs.
        actor_features = self._cat_partner_emb(actor_features, partner_emb)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, partner_emb=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        # Training path: re-evaluate stored actions. The embedding here must be
        # the SAME one used when the action was taken (passed down from the
        # buffer), or PPO's importance ratio is computed on mismatched inputs.
        actor_features = self._cat_partner_emb(actor_features, partner_emb)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values

    def get_policy_values(self, obs, rnn_states, masks):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        values = self.v_out(actor_features)

        return values

    def get_probs(self, obs, rnn_states, masks, available_actions=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_probs = self.act.get_probs(actor_features, available_actions)

        return action_probs, rnn_states

    def get_action_log_probs(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)

        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        if self._use_influence_policy:
            mlp_obs = self.mlp(obs)
            actor_features = torch.cat([actor_features, mlp_obs], dim=1)
        if self._layer_after_N > 0:
            actor_features = self.mlp_after(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )

        values = self.v_out(actor_features) if self._use_policy_vhead else None

        return action_log_probs, dist_entropy, values, rnn_states


class R_Critic(nn.Module):
    def __init__(self, args, share_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_influence_policy = args.use_influence_policy
        self._use_popart = args.use_popart
        self._influence_layer_N = args.influence_layer_N
        self._recurrent_N = args.recurrent_N
        self._layer_after_N = getattr(args, "layer_after_N", 0)
        self._num_v_out = getattr(args, "num_v_out", 1)
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        logger.trace(f"critic share obs shape: {share_obs_shape}")

        if "Dict" in share_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
        else:
            self._mixed_obs = False
            # MARK: MLPBase will not be used
            self.base = (
                CNNBase(args, share_obs_shape, cnn_layers_params=args.cnn_layers_params)
                if len(share_obs_shape) == 3
                else MLPBase(
                    args,
                    share_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=args.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                share_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        if self._layer_after_N > 0:
            self.mlp_after = MLPLayer(
                input_size,
                input_size,
                self._layer_after_N,
                self._use_orthogonal,
                self._activation_id,
            )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(input_size, self._num_v_out, device=device))
        else:
            self.v_out = init_(nn.Linear(input_size, self._num_v_out))

        self.to(device)

    def forward(self, share_obs, rnn_states, masks, task_id=None):
        if self._mixed_obs:
            for key in share_obs.keys():
                share_obs[key] = check(share_obs[key]).to(**self.tpdv)
        else:
            share_obs = check(share_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(share_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_share_obs = self.mlp(share_obs)
            critic_features = torch.cat([critic_features, mlp_share_obs], dim=1)

        if self._layer_after_N > 0:
            critic_features = self.mlp_after(critic_features)

        values = self.v_out(critic_features)

        if self._num_v_out > 1 and task_id is not None:
            assert len(task_id.shape) == len(values.shape) and np.prod(task_id.shape) * self._num_v_out == np.prod(
                values.shape
            ), (task_id.shape, values.shape)
            values = torch.gather(values, -1, task_id.long())

        return values, rnn_states
