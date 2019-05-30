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
  def __init__(self, path2files, experiments, image_names, Y, intensity_mean, intensity_var):
    assert len(image_names) == len(Y)
    assert len(experiments) == len(image_names)
    self.path = path2files
    self.experiments = experiments
    self.image_names = image_names
    self.Y = Y
    self.intensity_mean = intensity_mean
    self.intensity_var = intensity_var
  def __len__(self):
    return len(self.Y)
  def getXimage(self, index):
    img_name = self.image_names[index]
    experiment = self.experiments[index]
    img_loc = 'well_' + str(experiment) + '/' + img_name
    img_loc = os.path.join(self.path, img_loc)
    image = io.imread(img_loc)
    image = np.true_divide(color.rgb2gray(image) - self.intensity_mean, math.sqrt(self.intensity_var))
    image = np.reshape(image, newshape = (1, image.shape[0], image.shape[1]))  
    return torch.from_numpy(image).float()
#  def getXimage(self, index):
#    img_name = 'well' + str(self.well_labels[index]) + '_day' + str(self.day_label_X[index]) + '_well.png'
#    img_loc = os.path.join(self.path, img_name)
    # skimage.io.imread returns a numpy array
#    image = io.imread(img_loc)
    # convert to grey scale
#    image = np.true_divide(color.rgb2gray(image) - self.mean, self.sd)
    # add color axis because torch image: CxHxW
#    image = np.reshape(image, newshape = (1, image.shape[0], image.shape[1]))
#    return torch.from_numpy(image).float()
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
    larger_image = np.zeros((self.max_dim, self.max_dim))
    larger_image = np.pad(image, pad_width = ((0, self.max_dim - image.shape[0]), (0, self.max_dim - image.shape[1])), mode = 'constant', constant_values = 0.0)
    # resize and add color axis because torch image: CxHxW
    larger_image = np.reshape(larger_image, newshape = (1, self.max_dim, self.max_dim))
    # does it matter how the resizing is done?  I don't think so
    return torch.from_numpy(larger_image).float()
  def getY(self, index):
    Y = self.Y[index]
    return torch.from_numpy(np.asarray(self.Y[index], dtype=float)).float()
  def __getitem__(self, index):
    X = self.getAreaImage(index)
    y = self.getY(index)
    return X, y



class OrganoidMultipleDataset(data.Dataset):
    'dataset class for microwell organoid images'
    def __init__(self, path2files, image_names, Y, mean_sd_dict):
        for k, image_name in image_names.items():
            assert len(image_name) == len(Y)
        self.path = path2files
        self.image_names = image_names
        self.Y = Y
        self.mean_sd_dict = mean_sd_dict
    def __len__(self):
        return len(self.Y)
    def getXimage(self, index):
        all_images_list = []
        for day,img_names in self.image_names.items():
            #print(day, "   ", index)
            
            img_name = img_names[index]
            img_loc = os.path.join(self.path, img_name)
            image = io.imread(img_loc)
            mean, sd = self.mean_sd_dict[day]
            image = np.true_divide(color.rgb2gray(image) - mean, sd)
            all_images_list.append(image)
        images = np.array(all_images_list)
        return torch.from_numpy(images).float()
    def getY(self, index):
        Y = self.Y[index]
        return torch.from_numpy(np.asarray(self.Y[index], dtype=float)).float()
    def __getitem__(self, index):
        X = self.getXimage(index)
        y = self.getY(index)
        return X, y


