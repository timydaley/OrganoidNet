from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import sys
import pandas
from dataLoader import OrganoidDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 20, # low for testing
          'shuffle': True, 'num_workers' : 1}
max_epochs = 20
path = '/Users/Daley/Teaching/CS231N/CS231Nproject/CS231n_Tim_Shan_example_data/' # need to change

well_descriptions = pandas.read_csv('/Users/Daley/Teaching/CS231N/CS231Nproject/well_summary_A1_e0891BSA_all.csv', sep=',', header=0) # need to change name

#sizes = well_descriptions['mw_area shape'].tolist()
day0wells = well_descriptions[(well_descriptions['day'] == 0)]
day13wells = well_descriptions[(well_descriptions['day'] == 13)]
finalSizes = day13wells['mw_area shape']

well_labels = []
for i in range(4800):
  i2str = str(i)
  if len(i2str) == 1:
    i2str = '000' + i2str
  if len(i2str) == 2:
    i2str = '00' + i2str
  if len(i2str) == 3:
    i2str = '0' + i2str
  well_labels.append(i2str)

initial_train_set = OrganoidDataset(path2files = path, well_labels = well_labels, day_label_X = ['00']*4800, sizes = finalSizes)
training_generator = data.DataLoader(initial_train_set, **params)



# Convolutional neural network (two convolutional layers)
class SimpleConvNet(nn.Module):
  def __init__(self, in_channels = 4, layer1channels = 16, layer2channels = 16, out_size = 1):
    super(SimpleConvNet, self).__init__()
      self.layer1 = nn.Sequential(
                                  nn.Conv2d(in_channels, layer1channels,
                                            kernel_size=5, stride=1, padding=2),
                                  nn.BatchNorm2d(layer1channels),
                                  nn.ReLU()#,nn.MaxPool2d(kernel_size=2, stride=2)
                                  )
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(layer1channels, layer2channels,
                                              kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm2d(layer2channels),
                                    nn.ReLU()#,nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.fc = nn.Linear(in_features = 193*193*layer2channels, out_features = out_size)
    
    def forward(self, x):
      out = self.layer1(x)
      out = self.layer2(out)
      out = out.reshape(out.size(0), -1)
      out = self.fc(out)
      return out

in_channels = 4
out_size = 1
model = SimpleConvNet(in_channels = in_channels, out_size = out_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
loss = nn.MSELoss()
train_error_array = numpy.zeros(num_epochs)

# Loop over epochs
for epoch in range(max_epochs):
  # Training
  for local_batch, local_labels in training_generator:
    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
