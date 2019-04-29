from skimage import io, transform
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pandas
from dataLoader import OrganoidDataset

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 20, # low for testing
          'shuffle': True, 'num_workers' : 1}
max_epochs = 200
path = '/Users/Daley/Teaching/CS231N/CS231Nproject/CS231n_Tim_Shan_example_data/' # need to change
n_images = 2000
microwell_labels = range(n_images)
day_label_X = [0]*n_images

well_descriptions = pandas.read_csv('/Users/Daley/Teaching/CS231N/CS231Nproject/well_summary_A1_e0891BSA_all.csv', sep=',', header=0) # need to change name
sizes = well_descriptions['mw_area shape'].tolist()
