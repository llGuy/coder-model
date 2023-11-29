import model
import torch
from dataclasses import dataclass

import coder_model_sim as sim

@dataclass
class HyperParameters:
    timesteps_per_batch: int
    max_timesteps_per_episode: int
    delta: float

class ProximalPolicyOptimizer:
    def __init__(
        self,
        environment: sim.Environment,
        hparams: HyperParameters
    ):
        pass
