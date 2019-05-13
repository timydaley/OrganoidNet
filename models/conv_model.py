from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import sys
import pandas
from dataLoader import OrganoidDataset




# Convolutional neural network (two convolutional layers)
class SimpleConvNet(nn.Module):
  def __init__(self, in_channels = 1, layer1channels = 16, layer2channels = 16, out_size = 1):
    super(SimpleConvNet, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(in_channels, layer1channels,
                                          kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(layer1channels),
                                nn.ReLU())
    self.layer2 = nn.Sequential(nn.Conv2d(layer1channels, layer2channels,
                                          kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(layer2channels),
                                nn.ReLU())
    self.layer3 = nn.Sequential(nn.Conv2d(layer2channels, layer2channels,
                                          kernel_size=5, stride=1, padding=2),
                                nn.BatchNorm2d(layer2channels),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                )
    self.fc = nn.Linear(in_features = 193*193*layer2channels, out_features = out_size)
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out

class Flatten(nn.Module):
    def flatten(x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    def forward(self, x):
        return flatten(x)

class SimpleConvNet2():
    def __init__(self, in_channels = 1, layer1channels = 16, layer2channels = 16, out_size = 1):
        maxL_layer2channels = layer2channels/2
        super(SimpleConvNet2, self).__init__()
        model = nn.Sequential(
        nn.Conv2d(in_channels, layer1channels,kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(layer1channels),
        nn.ReLU(),

        nn.Conv2d(layer1channels, layer2channels,
                                              kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(layer2channels),
        nn.ReLU(),

        nn.Conv2d(layer2channels, layer2channels,kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(layer2channels),
        nn.ReLU(),
        nn.MaxPool2d(2),

        Flatten(),
        nn.Linear(in_features = maxL_layer2channels*193*193, out_features = out_size),
    )
    

