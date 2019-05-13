from skimage import io, transform, color
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, utils
import os
import math

class OrganoidDataset(data.Dataset):
  'dataset class for microwell organoid images'
  def __init__(self, path2files, well_labels, day_label_X, Y, intensity_mean = 0.5, intensity_var = 0.025):
    assert len(well_labels) == len(Y)
    assert len(day_label_X) == len(Y)
    self.path = path2files
    #self.mw_labels = microwell_labels
    self.well_labels = well_labels
    self.day_label_X = day_label_X
    self.Y = Y
    self.mean = intensity_mean
    self.sd = math.sqrt(intensity_var)
  def __len__(self):
    return len(self.Y)
  def getXimage(self, index):
    img_name = 'well' + str(self.well_labels[index]) + '_day' + str(self.day_label_X[index]) + '_well.png'
    img_loc = os.path.join(self.path, img_name)
    # skimage.io.imread returns a numpy array
    image = io.imread(img_loc)
    # convert to grey scale
    image = np.true_divide(color.rgb2gray(image) - self.mean, self.sd)
    # add color axis because torch image: CxHxW
    image = np.reshape(image, newshape = (1, image.shape[0], image.shape[1]))
    return torch.from_numpy(image).float()
  def getY(self, index):
    Y = self.Y[index]
    return torch.from_numpy(np.asarray(self.Y[index], dtype=float)).float()
  def __getitem__(self, index):
    X = self.getXimage(index)
    y = self.getY(index)
    return X, y

class OrganoidMwAreaDataset(data.Dataset):
  'dataset class for microwell area organoid images'
  def __init__(self, path2files, well_labels, day_label_X,
               Y, intensity_mean = 0.5, intensity_var = 0.025,
               max_dim = 132):
    assert len(well_labels) == len(Y)
    assert len(day_label_X) == len(Y)
    self.path = path2files
    #self.mw_labels = microwell_labels
    self.well_labels = well_labels
    self.day_label_X = day_label_X
    self.Y = Y
    self.mean = intensity_mean
    self.sd = math.sqrt(intensity_var)
    self.max_dim = max_dim
  def __len__(self):
    return len(self.Y)
  def getAreaImage(self, index):
    img_name = 'well' + str(self.well_labels[index]) + '_day' + str(self.day_label_X[index]) + '_mw_area.png'
    img_loc = os.path.join(self.path, img_name)
    # skimage.io.imread returns a numpy array
    image = io.imread(img_loc)
    # convert to grey scale
    image = np.true_divide(color.rgb2gray(image) - self.mean, self.sd)
    larger_image = np.zeros((max_dim, max_dim))
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        larger_image[i,j] = image[i,j]
    # resize and add color axis because torch image: CxHxW
    larger_image = np.reshape(larger_image, newshape = (1, self.max_dim, self.max_dim))
    # does it matter how the resizing is done?  I don't think so
    return torch.from_numpy(image).float()
  def getY(self, index):
    Y = self.Y[index]
    return torch.from_numpy(np.asarray(self.Y[index], dtype=float)).float()
  def __getitem__(self, index):
    X = self.getXimage(index)
    y = self.getY(index)
    return X, y




