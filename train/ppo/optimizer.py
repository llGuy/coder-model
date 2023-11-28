import env
import nets
import torch
from dataclasses import dataclass

@dataclass
class HyperParameters:
    timesteps_per_batch: int
    max_timesteps_per_episode: int
    delta: float

class ProximalPolicyOptimizer:
    def __init__(
        self,
        environment: env.Environment,
        hparams: HyperParameters
    ):
        self.env = environment
        self.hparams = hparams

        # Initialize policy network
        self.policy_net = MultiLayerNet(
            environment.observation_shape,
            [128, 64],
            environment.action_shape
        )

        # Initialize the baseline estimator
        self.base_estimator = MultiLayerNet(
            environment.observation_shape,
            [128, 64],
            1
        )

    def train(self, total_timesteps):
        num_timestep = 0
        while num_timestep < total_timesteps:
            pass

    def _rollout(self):
        batch_obs = []
        batch_acts = []
        batch_lprobs = []
        batch_rewards = []
        batch_rewards_tg = []
        batch_episode_lens = []

        t = 0

        while t < self.hparams.timesteps_per_batch:
            episode_rewards = []

            obs = self.env.reset()
            done = False

            for episode_t in range(self.hparams.max_timesteps_per_episode):
                t += 1
