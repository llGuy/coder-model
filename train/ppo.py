import model
import torch
import coder_model_sim as sim
from dataclasses import dataclass
from torch.distributions import MultivariateNormal

@dataclass
class HyperParameters:
    batch_size: int
    timesteps_per_batch: int
    max_timesteps_per_episode: int
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

    def learn(self, total_time_steps):
        # Keep track of how many steps in total have been simulated across batches
        global_num_timesteps = 0

        while global_num_timesteps < total_time_steps:
            batch_obs, batch_acts, batch_lprobs, batch_rewards_to_go, batch_lengths = self._rollout()

    def _rollout(self):
        batch_obs = []
        batch_acts = []
        batch_lprobs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_episode_lengths = []

        t = 0

        while t < self.hparams.timesteps_per_batch:
            episode_rewards = []

            self.env.reset()
            obs = self.env.get_prog_observations()

            for ep_t in range(self.hparams.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs)

                action, lprob = self._get_action(obs)

                self.env.step(action)

                obs = self.env.get_prog_observations()
                rewards = self.env.get_rewards()

                episode_rewards.append(rewards) # This has shape num_timesteps, batch_size
                batch_acts.append(action)
                batch_lprobs.append(lprob)

            batch_episode_lengths.append(ep_t + 1)
            batch_rewards.append(episode_rewards)

        batch_obs = torch.stack(batch_obs, dim=0)
        batch_acts = torch.stack(batch_acts, dim=0)
        batch_lprobs = torch.stack(batch_lprobs, dim=0)

        # batch_rewards starts off in the following format:
        #   list length num_episodes of
        #       list length timesteps_per_episode of
        #           tensor shape (batch_size,)
        # We want to turn this into a tensor of shape
        #   (batch_size, num_episodes, timesteps_per_episode)

        imm = []
        for ep_idx in range(len(batch_rewards)):
            imm.append(torch.stack(batch_rewards[ep_idx], dim=1))
        batch_rewards = torch.stack(imm).permute(1, 0, 2)

        # These are Q-values per timestep per batch.
        batch_rewards_to_go = self._compute_rewards_to_go(batch_rewards)

        return batch_obs, batch_acts, batch_lprobs, batch_rewards_to_go, batch_episode_lengths

    def _get_action(self, obs):
        mean = self.actor(obs, self.io_pair_obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()

    def _compute_rewards_to_go(self, batch_rewards):

        batch_size, num_episodes, timesteps_per_episode = batch_rewards.shape
        batched_rewards_to_go = torch.empty(batch_size, num_episodes, timesteps_per_episode)

        for ep_num in range(num_episodes):

            discounted_reward = torch.zeros(batch_size)

            for timestep in reversed(range(timesteps_per_episode)):
                discounted_reward += batch_rewards[:, ep_num, timestep] + \
                        self.hparams.gamma * discounted_reward

                batched_rewards_to_go[:, ep_num, timesteps_per_episode - timestep - 1] = \
                        discounted_reward.clone()

        return batched_rewards_to_go
