import model
import torch
from torch.optim import Adam
import torch.nn as nn
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
    clip: float
    lr: float

class ProximalPolicyOptimizer:
    def __init__(
        self,
        env: sim.SimManager,
        hparams: HyperParameters
    ):
        self.prog_obs_space = sim.prog_observation_size
        self.io_pair_obs_space = sim.io_pair_observation_size
        self.action_space = sim.action_size

        print(f"prog_obs_space={self.prog_obs_space}")
        print(f"io_pair_obs_space={self.io_pair_obs_space}")
        print(f"action_space={self.action_space}")

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

        self.actor_optim = Adam(self.actor.parameters(), lr=self.hparams.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.hparams.lr)
        

    def expand_io_obs(self, num_episodes, num_timesteps_per_episode):
        expanded_io_obs = self.io_pair_obs.unsqueeze(0)
        expanded_io_obs = expanded_io_obs.expand(num_timesteps_per_episode, -1, -1)
        expanded_io_obs = expanded_io_obs.unsqueeze(0)
        expanded_io_obs = expanded_io_obs.expand(num_episodes, -1, -1, -1)
        return expanded_io_obs

    """
    rollout_obs is of shape (num_episodes, num_timesteps_per_episode, batch_size, state_dim)
    """
    def evaluate(self, rollout_obs, rollout_acts):
        num_ep, num_timesteps_per_episode, batch_size, state_dim = rollout_obs.size()

        V = self.critic(rollout_obs, self.expand_io_obs(num_ep, num_timesteps_per_episode))
        mean = self.actor(rollout_obs, self.expand_io_obs(num_ep, num_timesteps_per_episode)) 

        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(rollout_acts)

        return V.squeeze(), log_probs

    def learn(self, total_time_steps):
        # Keep track of how many steps in total have been simulated across batches
        global_num_timesteps = 0

        while global_num_timesteps < total_time_steps:
            rollout_obs, rollout_acts, rollout_lprobs, rollout_rtgs = self._rollout()

            print(f"mean rewards to go: {rollout_rtgs.mean().item()}")

            V, _ = self.evaluate(rollout_obs, rollout_acts)
            A_k = rollout_rtgs - V.detach()

            # Normalize
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.hparams.num_epochs):
                V, cur_lprobs = self.evaluate(rollout_obs, rollout_acts)
                ratios = torch.exp(cur_lprobs - rollout_lprobs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.hparams.clip, 1 + self.hparams.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, rollout_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                print(f"actor loss: {actor_loss.item()}, critic loss {critic_loss.item()}")

            global_num_timesteps += self.hparams.num_episodes_per_rollout * \
                    self.hparams.num_timesteps_per_episode

    def _rollout(self):
        h = self.hparams

        # (num_episodes, timesteps_per_episode, batch_size, state_dim)
        rollout_obs = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.prog_obs_space
        )

        # (num_episodes, timesteps_per_episode, batch_size, act_dim)
        rollout_acts = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.action_space
        )

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_lprobs = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size
        )

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_rewards = torch.zeros(
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
        rollout_rewards_to_go = torch.zeros(num_episodes, timesteps_per_episode, batch_size)

        for ep_num in range(num_episodes):
            discounted_reward = torch.zeros(batch_size)

            for timestep in reversed(range(timesteps_per_episode)):
                discounted_reward += batch_rewards[ep_num, timestep] + \
                    self.hparams.gamma * discounted_reward

                rollout_rewards_to_go[ep_num, timestep] = discounted_reward

        return rollout_rewards_to_go
