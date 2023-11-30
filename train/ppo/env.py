import torch

# Encapsulates the environment logic and data which will be executed on the 
# C++ side.
#
# TODO: For now, the shapes are just placeholders.
class Environment:
    def __init__(self):
        # The observation is simply the current program. This is just going to
        # require 5 3-vectors where each 3-vector represents an instruction.
        self.observation_shape = (64)

        # Our action space is modification of an instruction.
        self.action_shape = (32)

    def reset():
        pass

    def step():
        pass
