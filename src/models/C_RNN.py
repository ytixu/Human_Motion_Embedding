import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import backend as K

import abs_model

class C_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		assert args['supervised']

		self.timesteps = args['timesteps']
		return super(C_RNN, self).__init__(args)

	def __create_conv_encoder(self):
		# The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
		# Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
		# the input timeseries, the activation of each filter at that position.
		conv1 = Convolution1D(nb_filter=64, filter_length=3, activation='relu', input_shape=(self.timesteps, self.input_dim))()
		# Downsample the output of convolution by 2X.
		pool1 = MaxPooling1D()
		conv2 = Convolution1D(nb_filter=16, filter_length=5, activation='relu')
		pool2 = MaxPooling1D()
		flat = Flatten()
		dense = Dense(self.output_dim)

		return conv1, pool1, conv2, pool2, flat, dense

	def make_model(self):
		self.input_dim = self.input_dim - self.name_dim
		self.output_dim = self.name_dim

		conv1, pool1, conv2, pool2, flat, dense = __create_conv_encoder()
		def conv_encoder(seq):
			return dense(flat(pool2(conv2(pool1(conv1(seq))))))

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = conv_encoder(inputs)
		decoded = K_layer.Lambda(lambda x: K.tf.nn.softmax(x))(decoded)
		output = K_layer.RepeatVector(self.timesteps)(decoded)

		self.model = Model(inputs, output)
		self.encoder = self.model
		self.decoder = self.model

	def load_embedding(self, data, **kwargs):
		# no embedding
		pass

	def format_data(self, x, **kwargs):
		return x[:,:,:-self.name_dim], x[:,:,-self.name_dim:]

	def predict(self, x, **kwargs):
		# not supported
		pass

	def classify(self, x, **kwargs):
		return self.model.predict(x)
