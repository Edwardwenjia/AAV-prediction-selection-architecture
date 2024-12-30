import math
import re
import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.callbacks import Callback
import warnings
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,GlobalAveragePooling2D, \
      concatenate, Activation, Conv2DTranspose, Flatten, Dense, Add, AveragePooling2D
from keras.applications import MobileNetV2
from keras.layers import Input, Dense, LayerNormalization, Dropout, Reshape, Permute, Embedding, Lambda, MultiHeadAttention, GlobalAveragePooling1D

#Modeling 
#------------------------------------------------
#%% ResNet model 
#------------------------------------------------

def resnet_block(input_data, filters, kernel_size=(3, 3), stride=1):
    """ Create a residual block """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(input_data)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    
    # Adjust dimensions with 1x1 convolution if input and output dimensions do not match
    shortcut = input_data
    if input_data.shape[-1] != filters or stride > 1:
        shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=stride)(input_data)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet_aa_model(input_shape=(7, 20, 1), L1=140, L2=20, learning_rate=0.0005):
    inputs = Input(input_shape)
    # Initial convolution layer
    x = Conv2D(filters=L1, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Residual block
    x = resnet_block(x, filters=L2)
    x = Flatten()(x)  # Flatten layer
    x = Dense(units=32, activation='relu')(x)  # Fully connected layer

    outputs = Dense(units=1)(x)  # Output layer
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


def resnet_seq_model(L1=420, L2=60):
    learning_rate = 0.0005
    inputs = Input(shape=(21, 20, 1))
    # Initial convolution layer
    x = Conv2D(filters=L1, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Residual block
    x = resnet_block(x, filters=L2)
    x = Flatten()(x)  # Flatten layer
    x = Dense(units=32, activation='relu')(x)  # Fully connected layer
    outputs = Dense(units=1)(x)  # Output layer
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model
