import torch
import coder_model_sim as sim

import ppo

data = torch.tensor([1, 2, 3], dtype=torch.float32)

print(f"Before {data}")

sim.inspect(data)

print(f"After {data}")

print(ppo.add(1, 2))
