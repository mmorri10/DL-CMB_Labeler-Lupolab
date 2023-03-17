import numpy as np

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import Flatten, Dense, Dropout, Activation, Add, concatenate
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow as tf


# ********************************************************************
# ********************************************************************
# 
# 							Building blocks
#
# ********************************************************************
# ********************************************************************

# original ResNet unit
def res_layer(x, width=16, first_layer=False):
	if first_layer:
		x_shortcut = Conv3D(width, (1, 1, 1), padding='same')(x)
	else:
		x_shortcut = x
	x = Conv3D(width, (3, 3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv3D(width, (3, 3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Add()([x, x_shortcut])
	x = Activation('relu')(x)
	return x

# ResNet unit with identity mapping
def res_layer_im(x, width=16, first_layer=False):
	if first_layer:
		x_shortcut = Conv3D(width, (1, 1, 1), padding='same')(x)
	else:
		x_shortcut = x
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv3D(width, (3, 3, 3), padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv3D(width, (3, 3, 3), padding='same')(x)
	x = Add()([x, x_shortcut])
	return x

def dense_block(x, n_conv, width):
	joint_input = x
	for i in range(n_conv):
		x = BatchNormalization()(joint_input)
		x = Activation('relu')(x)
		x = Conv3D(width, (3, 3, 3), padding='same')(x)
		joint_input = concatenate([joint_input, x])
	return x


# ********************************************************************
# ********************************************************************
# 
# 							   MODELS
#
# ********************************************************************
# ********************************************************************


# simple fully connected neural network, used as baseline
def fcnn(shape=[16, 16, 8], n_ch=1):
	model = Sequential()
	model.add(Flatten(input_shape=shape+[n_ch]))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	return model


# the model from 2016 cnn cmb paper
def cnn_2016(shape=[20, 20, 12], n_ch=1):
	inputs = Input(shape + [1])
	conv1 = Conv3D(32, (7, 7, 5), activation='relu')(inputs)
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	conv3 = Conv3D(64, (5, 5, 3), activation='relu')(pool2)

	flat = Flatten()(conv3)
	fc1 = Dense(500, activation='relu')(flat)
	fc2 = Dense(100, activation='relu')(fc1)
	output = Dense(1, activation='sigmoid')(fc2)

	model = Model(inputs=[inputs], outputs=[output])
	return model


# starting point, simple CNN setup
def simple_cnn(shape=[16, 16, 10], n_ch=1):
	model = Sequential()
	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=shape+[1]))
	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	# drop = Dropout(0.5)(pool3)

	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	# drop = Dropout(0.5)(pool6)

	model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	return model

# reduce the number of channels, simpler than simple, about 87k params
def simpler_cnn(shape=[16, 16, 8], n_ch=1):

	model = Sequential()
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=shape+[n_ch]))
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))

	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	return model

# 10/10
def simpler_cnn_v2(shape=[16, 16, 8], n_ch=1):

	model = Sequential()
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=shape+[n_ch]))
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))

	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	return model

# 10/11
# remove 2 conv layers, #params reduced to 29k
def simpler_cnn_v3(shape=[16, 16, 8], n_ch=1):

	model = Sequential()
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=shape+[n_ch]))
	model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(16, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	return model


def resnet_v5(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 8
	n_res = 4

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*2, first_layer=True)
	for i in range(1, n_res * 2):
		y = res_layer(y, width=res_width*2)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*4, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width*4)
	y = AveragePooling3D((4, 4, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model

def resnet_v4_1(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 4
	n_res = 3

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*2, first_layer=True)
	for i in range(1, n_res * 2):
		y = res_layer(y, width=res_width*2)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*4, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width*4)
	y = AveragePooling3D((4, 4, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model


def resnet_v4(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 8
	n_res = 3

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*2, first_layer=True)
	for i in range(1, n_res * 2):
		y = res_layer(y, width=res_width*2)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*4, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width*4)
	y = AveragePooling3D((4, 4, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model



# resnet v3: all residual unit, and deeper network
def resnet_v3(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 8
	n_res = 2

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*2, first_layer=True)
	for i in range(1, n_res * 2):
		y = res_layer(y, width=res_width*2)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	y = res_layer(y, width=res_width*4, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width*4)
	y = AveragePooling3D((4, 4, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model


# resnet v1: original resnet
def resnet_v1(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 8
	n_res = 5

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)

	# regular conv+pooling layers
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model


# resnet v2: original resnet
def resnet_v2(shape=[16, 16, 8], n_ch=1):

	# set network parameters
	res_width = 8
	n_res = 1

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width)

	y = res_layer(y, width=res_width*2, first_layer=True)
	for i in range(1, n_res):
		y = res_layer(y, width=res_width*2)

	# regular conv+pooling layers
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model

def resnetIm_v1(shape=[16, 16, 8], n_ch=1):
	res_width = 8
	n_res = 100

	# resisual layers
	x = Input(shape=shape+[n_ch])
	y = res_layer_im(x, width=res_width, first_layer=True)
	for i in range(1, n_res):
		y = res_layer_im(y, width=res_width)

	# regular conv+pooling layers
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(y)
	y = MaxPooling3D(pool_size=(2, 2, 2))(y)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)

	model = Model(input=x, output=y)
	return model



def densenet_v1(shape=[16,16,8], n_ch=1):
	x = Input(shape=shape+[n_ch])
	y = dense_block(x, 5, 4)
	y = Conv3D(4, (1, 1, 1), activation='relu', padding='same')(y)
	y = AveragePooling3D(pool_size=(2, 2, 2))(y)
	y = dense_block(y, 5, 8)
	y = Conv3D(8, (1, 1, 1), activation='relu', padding='same')(y)
	y = AveragePooling3D(pool_size=(2, 2, 2))(y)
	y = dense_block(y, 5, 16)

	# fc
	y = Flatten()(y)
	y = Dense(32, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(16, activation='relu')(y)
	y = Dropout(0.5)(y)
	y = Dense(1, activation='sigmoid')(y)	

	model = Model(input=x, output=y)
	return model

if __name__ == '__main__':
	model = resnet_v4(n_ch=1)
	print(model.summary())