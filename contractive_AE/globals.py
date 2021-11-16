import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf#; tf.compat.v1.disable_eager_execution()
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, \
    Conv3D, UpSampling3D, Conv3DTranspose, MaxPool3D, Activation, BatchNormalization, LeakyReLU, \
  Dropout, MaxPool2D, UpSampling2D, Lambda, AveragePooling2D, Add, AveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, SGD
import netCDF4 as nc
from keras.models import load_model
tf.executing_eagerly()

MODEL_SAVE_FOLDER = os.environ.get('MODEL_SAVE_FOLDER', 'models/')

def change_save_folder(newfolder):
    global MODEL_SAVE_FOLDER
    MODEL_SAVE_FOLDER = newfolder
