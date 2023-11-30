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
            layer_dims[512, 128],
            output_dim=self.action_space
        )

        # Produces an estimated baseline of value
        self.critic = model.MultiLayerNet(
            input_dim_prog=self.prog_obs_space,
            input_dim_io=self.io_pair_obs_space,
            post_input_dim=512,
            layer_dims[512, 128],
            output_dim=1
        )

        # This doesn't change - so just store it upon init
        self.io_pair_obs = self.env.get_io_pair_observations()

        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5)
        self.cot_mat = torch.diag(self.cov_var)

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

                episode_rewards.append(rewards)
                batch_acts.append(action)
                batch_lprobs.append(lprobs)

            batch_episode_lengths.append(epi_t + 1)
            batch_rewards.append(episode_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_lprobs = torch.tensor(batch_lprobs, dtype=torch.float)

        batch_rewards_to_go = self._compute_rewards_to_go(batch_rewards)

        return batch_obs, batch_acts, batch_lprobs, batch_rewards_to_go, batch_episode_lengths

    def _get_action(self, obs):
        mean = self.actor(obs, self.io_pair_obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def _compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []

        for ep_rewards in reversed(batch_rewards):
            discounted_reward = 0

            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rewards_to_go.insert(0, discounted_reward)

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)

        return batch_rewards_to_go

