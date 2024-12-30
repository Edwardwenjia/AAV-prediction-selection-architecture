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
#%% U-Net model 
#------------------------------------------------

def unet_block(input_data, filters):
    ''' 
    Create a standard convolutional block 
    '''
    x = Conv2D(filters, (3, 3), padding='same')(input_data)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    return x

def unet_aa_model(input_shape=(7, 20, 1)):
    learning_rate = 0.0005
    inputs = Input(shape=input_shape)
    # Encoder path (reduced down-sampling levels)
    down1 = unet_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)
    bottleneck = unet_block(pool1, 128) # Layer of bottleneck
    # Decoder path (corresponding to reduced down-sampling levels)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    concat1 = concatenate([up1, down1], axis=-1)
    up_conv1 = unet_block(concat1, 64)
    x = Flatten()(up_conv1)
    x = Dense(units=32, activation='relu')(x)
    outputs = Dense(units=1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


#------------------------------------------------
#%% U-Net model 
#------------------------------------------------

def mobilenetv2_aa_model(input_shape=(7,20,1), learning_rate=0.0005):
    inputs = Input(shape=input_shape)

    # Use pre-trained MobileNetV2 as a feature extractor, set the alpha parameter to reduce model size
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights=None, alpha=0.35, input_shape=input_shape)

    # Get the output of the last layer
    x = base_model.output

    # Since the input size is small, pooling layers may need adjustment to avoid excessively small output sizes
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(units=32, activation='relu')(x)

    # Output layer
    outputs = Dense(units=1)(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return model


