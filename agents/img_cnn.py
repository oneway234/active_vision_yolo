import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, in_channels=3):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(FC, self).__init__()
        self.bn1 = nn.BatchNorm1d(8112*39, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc2 = nn.Linear(8112*39, 512)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc2(x.view(x.size(0), -1)))
        return x