import os
import json
import model
import torch
import datetime
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
        self.cuda_supported = torch.cuda.is_available()
        self.cuda_device = -1
        if self.cuda_supported:
            print(f"CUDA is supported: device {torch.cuda.current_device()}")
            self.cuda_device = torch.cuda.current_device()

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
            layer_dims=[512, 256],
            output_dim=self.action_space,
            cuda_device = self.cuda_device
        )

        # Produces an estimated baseline of value
        self.critic = model.MultiLayerNet(
            input_dim_prog=self.prog_obs_space,
            input_dim_io=self.io_pair_obs_space,
            post_input_dim=512,
            layer_dims=[512, 256],
            output_dim=1,
            cuda_device = self.cuda_device
        )

        # This doesn't change - so just store it upon init
        self.io_pair_obs = self.env.get_io_pair_observations()

        if self.cuda_supported:
            self.io_pair_obs = self.io_pair_obs.to(self.cuda_device)
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()

        self.cov_var = torch.full(size=(self.action_space,), fill_value=1.5)
        self.cov_mat = torch.diag(self.cov_var).cuda()

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

        last_num_matches_mean = 0

        rollout_idx = 0

        while global_num_timesteps < total_time_steps:
            rollout_obs, rollout_acts, rollout_lprobs, rollout_rtgs = self._rollout()

            self.env.print_stats()

            matches_tensor = self.env.get_matches()

            last_num_matches_mean = matches_tensor.mean().item()
            rollout_rtgs_mean = rollout_rtgs.mean().item()
            print(f"Mean number of matches: {last_num_matches_mean}, mean rollouts to go: {rollout_rtgs_mean}")

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

            rollout_idx += 1
            global_num_timesteps += self.hparams.num_episodes_per_rollout * \
                    self.hparams.num_timesteps_per_episode

            if rollout_idx % 25 == 0:
                self._save(last_num_matches_mean, total_time_steps, True)

        self._save(last_num_matches_mean, total_time_steps, False)

    def _save(self, last_num_matches_mean, total_time_steps, tmp=True):
        models_path = os.path.dirname(__file__) + "/../models/"

        if not os.path.exists(models_path):
            os.makedirs(models_path)
            print("Made ../models directory")

        base_path = model.model_filename_base(last_num_matches_mean)
        if tmp:
            base_path = 'tmp'

        actor_path = models_path + "actor_" + base_path
        critic_path = models_path + "critic_" + base_path
        hp_path = models_path + "hyper_" + base_path + ".json"

        json_dict = {
            "batch_size": self.hparams.batch_size,
            "num_episodes_per_rollout": self.hparams.num_episodes_per_rollout,
            "num_timesteps_per_episode": self.hparams.num_timesteps_per_episode,
            "delta": self.hparams.delta,
            "gamma": self.hparams.gamma,
            "num_epochs": self.hparams.num_epochs,
            "clip": self.hparams.clip,
            "lr": self.hparams.lr,
            "total_timesteps": total_time_steps,
            "last_num_matches_mean": last_num_matches_mean
        }

        json_object = json.dumps(json_dict, indent=4)

        torch.save(self.actor, actor_path)
        torch.save(self.critic, critic_path)

        with open(hp_path, "w") as outfile:
            outfile.write(json_object)

    def _rollout(self):
        h = self.hparams

        # (num_episodes, timesteps_per_episode, batch_size, state_dim)
        rollout_obs = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.prog_obs_space
        ).cuda()

        # (num_episodes, timesteps_per_episode, batch_size, act_dim)
        rollout_acts = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size,
            self.action_space
        ).cuda()

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_lprobs = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size
        ).cuda()

        # (num_episodes, timesteps_per_episode, batch_size)
        rollout_rewards = torch.zeros(
            h.num_episodes_per_rollout,
            h.num_timesteps_per_episode,
            h.batch_size
        ).cuda()

        for episode_idx in range(h.num_episodes_per_rollout):
            self.env.reset()

            # (batch_size, state_dim)
            obs = self.env.get_prog_observations()
            if self.cuda_supported:
                obs = obs.to(self.cuda_device)

            for timestep_idx in range(h.num_timesteps_per_episode):
                # batch_obs.append(obs)
                rollout_obs[episode_idx, timestep_idx] = obs

                action, lprob = self._get_action(obs)
                self.env.step(action.cpu())

                # (batch_size, state_dim)
                obs = self.env.get_prog_observations().cuda()
                
                # (batch_size, 1)
                rewards = self.env.get_rewards().cuda()

                """
                if self.cuda_supported:
                    obs = obs.to(self.cuda_device)
                    rewards = rewards.to(self.cuda_device)
                """

                rollout_rewards[episode_idx, timestep_idx] = rewards
                rollout_acts[episode_idx, timestep_idx] = action.cuda()
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
        if self.cuda_supported:
            rollout_rewards_to_go = rollout_rewards_to_go.to(self.cuda_device)

        for ep_num in range(num_episodes):
            discounted_reward = torch.zeros(batch_size).cuda()

            for timestep in reversed(range(timesteps_per_episode)):
                discounted_reward += batch_rewards[ep_num, timestep] + \
                    self.hparams.gamma * discounted_reward

                rollout_rewards_to_go[ep_num, timestep] = discounted_reward

        return rollout_rewards_to_go
