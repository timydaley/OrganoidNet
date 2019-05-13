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

params = {'batch_size': 100, # low for testing
  'shuffle': True, 'num_workers' : 1}

max_epochs = 200

path = '/Users/Daley/Teaching/CS231N/CS231Nproject/CS231n_Tim_Shan_example_data/' # need to change

well_descriptions = pandas.read_csv('processed_well_descriptions.txt', sep='\t', header=0)
#sizes = well_descriptions['mw_area shape'].tolist()
day1wells = well_descriptions[(well_descriptions['day'] == 1)]
day13wells = well_descriptions[(well_descriptions['day'] == 13)]
finalSizes = day13wells['normalized hyst2 area'].values

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

initial_train_set = OrganoidDataset(path2files = path, well_labels = well_labels[0:1000], day_label_X = ['01']*1000, sizes = finalSizes[0:1000])
training_generator = data.DataLoader(initial_train_set, **params)


in_channels = 1
out_size = 1
model = SimpleConvNet(in_channels = in_channels, out_size = out_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=10**-8, weight_decay=0)
loss = nn.MSELoss()
train_error_array = np.zeros(max_epochs)



# Loop over epochs
for epoch in range(max_epochs):
  # Training
  print(epoch)
  optimizer.zero_grad()
  totalbatchMSE = 0.0
  for local_X, local_Y in training_generator:
    local_X, local_Y = local_X.to(device), local_Y.to(device)
    Y_hat = model.forward(local_X)
    train_error = loss(Y_hat, local_Y)
    train_error.backward()
    optimizer.step()
    model.eval() # set evaluation mode
    Y_hat = model.forward(local_X)
    train_error = loss(Y_hat, local_Y).item()
    totalbatchMSE = totalbatchMSE + params['batch_size']*train_error/4800 # rescale train_error, since MSE = \sum sqrt(|Y - hat(Y)|^2) / batch_size
    train_error_array[epoch] = totalbatchMSE
    np.savetxt(fname = "train_error_array_lr0.01_3convLayers.txt", X = train_error_array[range(epoch)])

