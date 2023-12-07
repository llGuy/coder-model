import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime

# This defines a simple multi-layer network which takes in two inputs:
# - the program observations
# - the io pair observations
class MultiLayerNet(nn.Module):
    # input_dim_prog: dimensions of the tensor containing the programs
    # input_dim_io: dimensions of the tensor containing the IO pairs
    # post_input_dim: the dimensions of the hidden layer which
    #                 connects the prog and io into a single tensor
    def __init__(
        self, 
        input_dim_prog: int,
        input_dim_io: int,
        post_input_dim: int,
        layer_dims: list[int],
        output_dim: int,
        cuda_device = -1
    ):
        super(MultiLayerNet, self).__init__()

        self.prog_fc = nn.Linear(input_dim_prog, post_input_dim).cuda()
        self.io_fc = nn.Linear(input_dim_io, post_input_dim).cuda()

        self.fc = []

        prev_dim = post_input_dim * 2

        for hidden_dim in layer_dims:
            self.fc.append(nn.Linear(prev_dim, hidden_dim).cuda())
            prev_dim = hidden_dim

        self.fc.append(nn.Linear(prev_dim, output_dim).cuda())

        self.cuda_device = cuda_device

    def forward(self, x_prog, x_io_pair):
        a_prog = F.relu(self.prog_fc(x_prog))
        a_io = F.relu(self.io_fc(x_io_pair))

        # Merge the a_prog and a_io tensors
        merged = torch.cat((a_prog, a_io), dim=len(x_prog.size())-1)

        if self.cuda_device != -1:
            merged = merged.to(self.cuda_device)

        for i in range(len(self.fc)):
            merged = self.fc[i](merged)

            # Only apply non-linearity on layers before the last
            if i != (len(self.fc) - 1):
                merged = F.relu(merged)

        return merged


# Prefix 'actor_', 'critic_', 'hyper_' when actually saving
def model_filename_base(mean_reward_to_go):
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    # file_name = f"../models/model_{formatted_datetime}_loss_{final_loss}"
    file_name = f"{mean_reward_to_go}_{formatted_datetime}"
    return file_name
