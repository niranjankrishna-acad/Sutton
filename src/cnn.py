import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape, action_space):
        super(CNN, self).__init__()
        
        self.networkConv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            
        )
        self.networkLinear = nn.Sequential(
            nn.Linear(self.get_feature_size(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        x = self.networkConv(x / 255.0)
        return self.networkLinear(x)

    def get_feature_size(self, input_shape):
        return self.networkConv(torch.zeros(1,*input_shape)).view(-1,1).size(dim=0)
