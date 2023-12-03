import model
import torch
import coder_model_sim as sim
from dataclasses import dataclass
from torch.distributions import MultivariateNormal

@dataclass
class HyperParameters:
    batch_size: int
    num_episodes_per_rollout: int
    num_timesteps_per_episode: int
    delta: float
    gamma: float
    num_epochs: int

class ProximalPolicyOptimizer:
    def __init__(
        self,
        env: sim.SimManager,
        hparams: HyperParameters
    ):
        self.prog_obs_space = sim.prog_observation_size
        self.io_pair_obs_space = sim.io_pair_observation_size
        self.action_space = sim.action_size

        self.env = env
        self.hparams = hparams

        # Produces actions
        self.actor = model.MultiLayerNet(
            input_dim_prog=self.prog_obs_space,
            input_dim_io=self.io_pair_obs_space,
            post_input_dim=512,
            layer_dims=[512, 128],
            output_dim=self.action_space
        )

        # Produces an estimated baseline of value
        self.critic = model.MultiLayerNet(
            input_dim_prog=self.prog_obs_space,
            input_dim_io=self.io_pair_obs_space,
            post_input_dim=512,
            layer_dims=[512, 128],
            output_dim=1
        )

        # This doesn't change - so just store it upon init
        self.io_pair_obs = self.env.get_io_pair_observations()

        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    """
    batched_rollouts is of shape (num_episodes, batch_size, timesteps_per_episode).
    """
    def evaluate(self, batch_obs):
        num_ep, batch_size, timesteps_per_episode = batch_obs.size()
        
        pass    


    def learn(self, total_time_steps):
        # Keep track of how many steps in total have been simulated across batches
        global_num_timesteps = 0

        while global_num_timesteps < total_time_steps:
            rollout_obs, rollout_acts, rollout_lprobs, rollout_rewards_to_go = self._rollout()

    def _rollout(self):
        h = self.hparams

        # (num_episodes, timesteps_per_episode, batch_size, state_dim)
        rollout_obs = torch.empty(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.prog_obs_space
        )

        # (num_episodes, timesteps_per_episode, batch_size, act_dim)
        rollout_acts = torch.empty(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.action_space
        )

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_lprobs = torch.empty(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size
        )

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_rewards = torch.empty(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size
        )

        for episode_idx in range(h.num_episodes_per_rollout):
            self.env.reset()

            # (batch_size, state_dim)
            obs = self.env.get_prog_observations()

            for timestep_idx in range(h.num_timesteps_per_episode):
                # batch_obs.append(obs)
                rollout_obs[episode_idx, timestep_idx] = obs

                action, lprob = self._get_action(obs)
                self.env.step(action)

                # (batch_size, state_dim)
                obs = self.env.get_prog_observations()

                # (batch_size, 1)
                rewards = self.env.get_rewards()

                rollout_rewards[episode_idx, timestep_idx] = rewards
                rollout_acts[episode_idx, timestep_idx] = action
                rollout_lprobs[episode_idx, timestep_idx] = lprob

        # These are Q-values per timestep per batch.
        rollout_rewards_to_go = self._compute_rewards_to_go(rollout_rewards)

        return rollout_obs, rollout_acts, rollout_lprobs, rollout_rewards_to_go

    def _get_action(self, obs):
        mean = self.actor(obs, self.io_pair_obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

    def _compute_rewards_to_go(self, batch_rewards):

        num_episodes, timesteps_per_episode, batch_size = batch_rewards.shape
        rollout_rewards_to_go = torch.empty(num_episodes, timesteps_per_episode, batch_size)

        for ep_num in range(num_episodes):
            discounted_reward = torch.zeros(batch_size)

            for timestep in reversed(range(timesteps_per_episode)):
                discounted_reward += batch_rewards[ep_num, timestep] + \
                    self.hparams.gamma * discounted_reward

                rollout_rewards_to_go[ep_num, timestep] = discounted_reward.clone()

        return rollout_rewards_to_go
