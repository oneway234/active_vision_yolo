import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions=7):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(316372, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)