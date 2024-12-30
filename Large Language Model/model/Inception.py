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
#%% Inception model 
#------------------------------------------------


def inception_module(input_data, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    """Creates an Inception module."""
    # 1x1 Convolution branch
    branch_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), activation='relu', padding='same')(input_data)

    # 3x3 Convolution branch
    branch_3x3 = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), activation='relu', padding='same')(input_data)
    branch_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), activation='relu', padding='same')(branch_3x3)

    # 5x5 Convolution branch
    branch_5x5 = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), activation='relu', padding='same')(input_data)
    branch_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), activation='relu', padding='same')(branch_5x5)

    # Pooling branch
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_data)
    branch_pool = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), activation='relu', padding='same')(branch_pool)

    # Concatenate all branches
    output = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool], axis=-1)
    return output


def inception_aa_model(input_shape, L1=140, L2=20, learning_rate=0.0005):
    """
    Builds an Inception-based model for regression tasks.
    
    Args:
        input_shape (tuple): Shape of the input data.
        L1 (int): Number of filters for the initial Conv2D layer.
        L2 (int): Number of units for a fully connected dense layer.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.Model: Compiled Inception model.
    """
    # Input layer
    inputs = Input(input_shape)

    # Initial convolution and pooling layers
    x = Conv2D(filters=L1, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Inception module
    x = inception_module(
        x, 
        filters_1x1=64, 
        filters_3x3_reduce=96, 
        filters_3x3=128, 
        filters_5x5_reduce=16, 
        filters_5x5=32, 
        filters_pool_proj=32
    )

    # Flatten the feature maps
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(units=32, activation='relu')(x)  # Intermediate dense layer
    outputs = Dense(units=1)(x)  # Output layer for regression

    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


