from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import sys
import pandas
import math
from dataLoader import OrganoidDataset

# Convolutional neural network (two convolutional layers)                                                                                                                                                
class SimpleConvNetRegression(nn.Module):
  def __init__(self, in_channels = 1, layer1channels = 512, layer2channels = 256, out_size = 1, in_size = 193):
    super(SimpleConvNetRegression, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(in_channels, layer1channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(layer1channels),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
    layer1dim = math.floor((in_size -1 - 1)/2) + 1
    self.layer2 = nn.Sequential(nn.Conv2d(layer1channels, layer2channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(layer2channels),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
    layer2dim = math.floor((layer1dim -1 - 1)/2) + 1
    self.layer3 = nn.Sequential(nn.Conv2d(layer2channels, layer2channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(layer2channels),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
    layer3dim = math.floor((layer2dim -1 - 1)/2) + 1
    self.pre_fc = nn.Linear(in_features = layer3dim*layer3dim*layer2channels, out_features = layer3dim*layer3dim*layer2channels)
    self.final_fc = nn.Linear(in_features = layer3dim*layer3dim*layer2channels, out_features = out_size)
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.reshape(out.size(0), -1)
    out = self.pre_fc(out)
    out = nn.functional.relu(out)
    out = self.final_fc(out)
    return out

# Convolutional neural network (two convolutional layers)                                                                                                                                                
class SimpleConvNetClassification(nn.Module):
  def __init__(self, in_channels = 1, layer1channels = 64, layer2channels = 32, layer3channels = 16, out_size = 1, in_size = 193):
    super(SimpleConvNetClassification, self).__init__()
    self.layer1 = nn.Sequential(nn.Conv2d(in_channels, layer1channels,
                                          kernel_size=2, stride=1, padding=0),
                                nn.BatchNorm2d(layer1channels),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2))
    layer1dim = math.floor((in_size + 2 *0 - 1*(2 - 1) - 1)/1) + 1 # size after convs                                                                                                                 
    layer1dim = math.floor((layer1dim + 2*0 - 1*(2 - 1) - 1)/2) + 1 # size after max pool     
    self.layer2 = nn.Sequential(nn.Conv2d(layer1channels, layer2channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(layer2channels),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2))
    layer2dim = math.floor((layer1dim + 2 *1 - 1*(3 - 1) - 1)/1) + 1 # size after convs
    layer2dim = math.floor((layer2dim + 2*0 - 1*(2 - 1) - 1)/2) + 1 # size after max pool
    self.layer3 = nn.Sequential(nn.Conv2d(layer2channels, layer3channels,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(layer3channels),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size = 2))
    layer3dim = math.floor((layer2dim + 2 *1 - 1*(3 - 1) - 1)/1) + 1 # size after convs                                                                                                                
    layer3dim = math.floor((layer3dim + 2*0 - 1*(2 - 1) - 1)/2) + 1 # size after max pool   
    self.pre_fc = nn.Linear(in_features = layer3dim*layer3dim*layer3channels, out_features = layer3dim*layer3dim*layer3channels)
    self.final_fc = nn.Linear(in_features = layer3dim*layer3dim*layer3channels, out_features = out_size)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = out.view(out.size()[0], -1)
    out = self.pre_fc(out)
    out = nn.functional.relu(out)
    out = self.final_fc(out)
    out = out.sigmoid()
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
    

