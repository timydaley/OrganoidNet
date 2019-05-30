from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import sys
import pandas
from conv_model import SimpleConvNetClassification
from dataLoader import OrganoidDataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 11, # 3883/11 = 353
  'shuffle': True, 'num_workers' : 1}

max_epochs = 500

path = '../data/14day/'
# get ids
A2_C1_filtered_train = pandas.read_csv('A2_C1_filtered_train.csv', header = 0)

train_day2_mean_and_var = pandas.read_csv("train_day2_mean_and_var.txt", header = 0, delimiter = '\t')

train_set = OrganoidDataset(path2files = path, experiments = A2_C1_filtered_train['condition'], image_names = A2_C1_filtered_train['image_name_2'], Y = A2_C1_filtered_train['has_cell_13'], intensity_mean = train_day2_mean_and_var['mean'][0], intensity_var = train_day2_mean_and_var['variance'][0])
training_generator = data.DataLoader(train_set, **params)

A2_C1_filtered_validation = pandas.read_csv('A2_C1_filtered_validation.csv', header = 0)
n_val = A2_C1_filtered_validation.shape[0]
validation_set = OrganoidDataset(path2files = path, experiments = A2_C1_filtered_validation['condition'], image_names = A2_C1_filtered_validation['image_name_2'], Y = A2_C1_filtered_validation['has_cell_13'], intensity_mean = train_day2_mean_and_var['mean'][0], intensity_var = train_day2_mean_and_var['variance'][0])
val_params = {'batch_size': 1, 'shuffle': True, 'num_workers' : 1}
validation_generator = data.DataLoader(validation_set, **val_params)


in_channels = 1
out_size = 1
model = SimpleConvNetClassification(in_channels = in_channels, layer1channels = 64, layer2channels = 32, layer3channels = 16).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()
avg_error_array = np.zeros(max_epochs)
batch_error_array = np.zeros(max_epochs)
validation_error_array = np.zeros(max_epochs)

# Loop over epochs
for epoch in range(max_epochs):
  # Training
  print(epoch)
  batchLoss = 0.0
  avgLoss = 0.0
  valLoss = 0.0
  batch = 0
  print('begin training')
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
    batchLoss = train_error
    avgLoss = avgLoss + train_error/353
  avg_error_array[epoch] = avgLoss
  batch_error_array[epoch] = batchLoss
  model.eval()
  for local_X, local_Y in validation_generator:
    local_X, local_Y = local_X.to(device), local_Y.to(device)
    Y_hat = model.forward(local_X)
    val_error = loss(Y_hat, local_Y).item()
    valLoss = valLoss + val_error/n_val
  validation_error_array[epoch] = valLoss
  np.savetxt(fname = "../LogisticResults/A2_C1_day2_avg_error_lr0.001_3kern_64channels.txt", X = avg_error_array[range(epoch + 1)])
  np.savetxt(fname = "../LogisticResults/A2_C1_day2_batch_error_lr0.001_3kern_64channels.txt", X = batch_error_array[range(epoch + 1)])
  np.savetxt(fname = "../LogisticResults/A2_C1_day2_val_error_lr0.001_3kern_64channels.txt", X = validation_error_array[range(epoch + 1)])
  if epoch % 5 == 1:
    torch.save(model.state_dict(), 'saved_models/last_trained_model_lr0.001_3kern_64channels')
