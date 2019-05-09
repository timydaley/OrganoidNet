from skimage import io, transform, color
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, utils
import os

class OrganoidDataset(data.Dataset):
  'dataset class for microwell organoid images'
  def __init__(self, path2files, well_labels, day_label_X, sizes):
    assert len(well_labels) == len(sizes)
    assert len(day_label_X) == len(sizes)
    self.path = path2files
    #self.mw_labels = microwell_labels
    self.well_labels = well_labels
    self.day_label_X = day_label_X
    self.sizes = sizes
  def __len__(self):
    return len(self.sizes)
  def getXimage(self, index):
    img_name = 'well' + str(self.well_labels[index]) + '_day' + str(self.day_label_X[index]) + '_well.png'
    img_loc = os.path.join(self.path, img_name)
    # skimage.io.imread returns a numpy array
    image = io.imread(img_loc)
    # convert to grey scale
    image = color.rgb2gray(image)
    # add color axis because torch image: CxHxW
    image = np.reshape(image, newshape = (1, image.shape[0], image.shape[1]))
    return torch.from_numpy(image).float()
  def getY(self, index):
    Y = self.sizes[index]
    return torch.from_numpy(np.asarray(self.sizes[index], dtype=float)).float()
  def __getitem__(self, index):
    X = self.getXimage(index)
    y = self.getY(index)
    return X, y
  

  

