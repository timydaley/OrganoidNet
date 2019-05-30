from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import sys
import pandas
from conv_model import SimpleConvNet
from dataLoader import OrganoidMwAreaDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 22, # 4642/22 = 211
  'shuffle': True, 'num_workers' : 1}

max_epochs = 500

path = '../data/CS231n_Tim_Shan_example_data/'
well_descriptions = pandas.read_csv('filtered_well_descriptions.txt', header=0)
day1wells = well_descriptions['well_id']
day1wells = day1wells[well_descriptions['day'] == 1]
day1wells.shape
day13wells = well_descriptions['well_id']
day13wells = day13wells[well_descriptions['day'] == 13]
day13wells.shape
daysLabel = pandas.Series(list(set(day13wells) & set(day1wells)))


well_labels = []
for i in range(daysLabel.shape[0]):
  i2str = str(daysLabel[i])
  if len(i2str) == 1:
    i2str = '000' + i2str
  if len(i2str) == 2:
    i2str = '00' + i2str
  if len(i2str) == 3:
    i2str = '0' + i2str
  well_labels.append(i2str)

day_label_X = ['13']*len(well_labels)
n = len(well_labels)

finalSizes = well_descriptions['hyst2_area']
finalSizes = finalSizes[np.logical_and(well_descriptions['day'] == 13, np.isin(well_descriptions['well_id'], daysLabel))].values

day13_mean_and_var = pandas.read_csv('day13_mw_area_mean_and_var.txt', sep = '\t', header = 0)

initial_train_set = OrganoidMwAreaDataset(path2files = path, well_labels = well_labels, day_label_X = day_label_X, Y = finalSizes, intensity_mean = day13_mean_and_var['mean'][0], intensity_var = day13_mean_and_var['variance'][0], max_dim = 132)
training_generator = data.DataLoader(initial_train_set, **params)


in_channels = 1
out_size = 1
model = SimpleConvNet(in_channels = in_channels, out_size = out_size, in_size = 132, layer1channels = 256, layer2channels = 128).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
loss = nn.MSELoss()
batch_error_array = np.zeros(max_epochs)
avg_error_array = np.zeros(max_epochs)


# Loop over epochs
for epoch in range(max_epochs):
  # Training
  print("epoch: ", epoch)
  batchMSE = 0.0
  avgMSE = 0.0
  batch = 0
  for local_X, local_Y in training_generator:
    local_X, local_Y = local_X.to(device), local_Y.to(device)
    optimizer.zero_grad()
    Y_hat = model.forward(local_X)
    train_error = loss(Y_hat, local_Y)
    train_error.backward()
    optimizer.step()
    model.eval() # set evaluation mode
    Y_hat = model.forward(local_X)
    train_error = loss(Y_hat, local_Y).item()
    batchMSE = train_error
    avgMSE = avgMSE + train_error/211
  batch_error_array[epoch] = batchMSE
  avg_error_array[epoch] = avgMSE
  np.savetxt(fname = "mw_area_day13_batch_error_lr3eminus4_extra_fc_layer_more_channels.txt", X = batch_error_array[range(epoch + 1)])
  np.savetxt(fname = "mw_area_day13_avg_error_lr3eminus4_extra_fc_layer_more_channels.txt", X = avg_error_array[range(epoch + 1)])

