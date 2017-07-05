from keras.utils.visualize_util import plot
from keras.models import load_model
import scipy.misc as misc
import numpy as np
import os, pdb


model = load_model('model.h5')
plot(model, to_file='images/model.png')

img = misc.imread('images/center_2016_12_01_13_45_57_291.jpg')
# pdb.set_trace()
misc.imsave('images/flip_center_2016_12_01_13_45_57_291.jpg', np.fliplr(img))