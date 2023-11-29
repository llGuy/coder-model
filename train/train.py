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

    print(f"Program observation size: {sim.prog_observation_size}")
    print(f"IO pair observation size: {sim.io_pair_observation_size}")
    print(f"Action size: {sim.action_size}");

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If run==train
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()

    params = TrainParams(args.batch_size)

    main(params)
