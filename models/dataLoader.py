from skimage import io, transform
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, utils
import os

class OrganoidDataset(data.Dataset):
  'dataset class for microwell organoid images'
  def __init__(self, path2files, microwell_labels, day_label_X, sizes):
    assert len(microwell_labels) == len(well_labels) && len(well_labels) == len(sizes) && len(day_label_X) == len(sizes)
    self.path = path2files
    self.mw_labels = microwell_labels
    self.day_label_X = day_label_X
    self.sizes = sizes
  def __len__(self):
    return self.well_label
  def __getXimage__(self, index):
    img_name = 'well' + str(self.microwell_label[index]) + '_day' + str(self.day_label_X[index]) + '_well.png'
    img_loc = os.path.join(self.path, img_name)
    # skimage.io.imread returns a numpy array
    image = io.imread(img_loc)
    # swap color axis because numpy image: HxWxC but torch image: CxHxW                                                                                                                          
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image)
  def __getYsize__(self, index):
    return self.sizes[indx]
  def __getitem__(self, index):
    X = self.getXimage(index)
    y = self.getYsize(index)
    return X, y
  

  

