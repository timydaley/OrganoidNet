from skimage import io, transform, color
import numpy as np
import sys
import os
import pandas

def update_mean(x, x_bar_prev, n):
  return x_bar_prev + (x - x_bar_prev)/n

def update_sum_square_diff(x, ssd_prev, x_bar_prev, x_bar_curr):
  return ssd_prev + (x - x_bar_prev)*(x - x_bar_curr)



#assert len(sys.argv) == 3
#path = sys.argv[1]
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

x_bar_curr = 0.0
x_bar_prev = 0.0
ssd_curr = 0.0
n = 0
for index in range(len(well_labels)):
  print(index)
  img_name = 'well' + str(well_labels[index]) + '_day' + str(day_label_X[index]) + '_mw_area.png'
  img_loc = os.path.join(path, img_name)
  # skimage.io.imread returns a numpy array
  image = io.imread(img_loc)
  # convert to grey scale
  image = color.rgb2gray(image)
  for i, x in np.ndenumerate(image):
    n = n + 1
    x_bar_prev = x_bar_curr
    x_bar_curr = update_mean(x, x_bar_prev, n)
    ssd_curr = update_sum_square_diff(x, ssd_curr, x_bar_prev, x_bar_curr)

filname = "day13_mw_area_mean_and_var.txt"
with open(filname, "w") as f:
  print("mean\tvariance", file = f)
  print(x_bar_curr, '\t', ssd_curr/n, file = f)


