import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf#; tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, \
  Dropout, MaxPool2D, UpSampling2D, Lambda, AveragePooling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
import netCDF4 as nc
from keras.models import load_model
tf.executing_eagerly()
