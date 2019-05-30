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
path = '../data/14day/'
A2_C1_filtered_train = pandas.read_csv('A2_C1_filtered_train.csv', header = 0)
for day in range(14):
  print(day)
  x_bar_curr = 0.0
  x_bar_prev = 0.0
  ssd_curr = 0.0
  n = 0
  for index in range(A2_C1_filtered_train.shape[0]):
    day_name = 'image_name_' + str(day)
    img_name = A2_C1_filtered_train[day_name][index]
    img_loc = 'well_' + str(A2_C1_filtered_train['condition'][index]) + '/' + img_name
    img_loc = os.path.join(path, img_loc)
  # skimage.io.imread returns a numpy array
    image = io.imread(img_loc)
    # convert to grey scale
    image = color.rgb2gray(image)
    for i, x in np.ndenumerate(image):
      n = n + 1
      x_bar_prev = x_bar_curr
      x_bar_curr = update_mean(x, x_bar_prev, n)
      ssd_curr = update_sum_square_diff(x, ssd_curr, x_bar_prev, x_bar_curr)

  filname = 'train_day' + str(day) + '_mean_and_var.txt'
  with open(filname, "w") as f:
    print("mean\tvariance", file = f)
    print(x_bar_curr, '\t', ssd_curr/n, file = f)


