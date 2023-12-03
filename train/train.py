import ppo
import torch
import argparse
import coder_model_sim as sim
from dataclasses import dataclass

@dataclass
class TrainParams:
    batch_size: int

def main(params: TrainParams):
    # Create the simulation manager
    env = sim.SimManager(params.batch_size)

    # Sanity check that the module works
    print(f"Program observation size: {sim.prog_observation_size}")
    print(f"IO pair observation size: {sim.io_pair_observation_size}")
    print(f"Action size: {sim.action_size}");

    # Placeholder values - each rollout will have a total of 30 timesteps
    hparams = ppo.HyperParameters(
        batch_size=params.batch_size,
        num_episodes_per_rollout=2,
        num_timesteps_per_episode=15,
        delta=0.1,
        gamma=0.2,
        num_epochs=5,
        clip=0.2,
        lr=0.001
    )

    optimizer = ppo.ProximalPolicyOptimizer(env, hparams)

    optimizer.learn(1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If run==train
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    params = TrainParams(args.batch_size)

    main(params)
