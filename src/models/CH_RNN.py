import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import backend as K

import HH_RNN

class CH_RNN(HH_RNN.HH_RNN):

	def __create_conv_encoder(self, nb_filter, filter_length):
		# The first conv layer learns `nb_filter` filters (aka kernels), each of size ``(filter_length, nb_input_series)``.
		# Its output will have shape (None, window_size - filter_length + 1, nb_filter), i.e., for each position in
		# the input timeseries, the activation of each filter at that position.
		conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu', input_shape=(self.unit_t, self.input_dim))()
		# Downsample the output of convolution by 2X.
		pool1 = MaxPooling1D()
		conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation='relu')
		pool2 = MaxPooling1D()
		flat = Flatten()
		# For binary classification, change the activation to 'sigmoid'
		dense = Dense(self.latent_dim/2, activation=self.activation)

		return conv1, pool1, conv2, pool2, flat, dense

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		# encode_1 = abs_model.RNN_UNIT(self.latent_dim/2)
		encode_2 = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)

		conv1, pool1, conv2, pool2, flat, dense = __create_conv_encoder(64, 3)
		def conv_encoder(seq):
			return dense(flat(pool2(conv2(pool1(conv1(seq))))))

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = conv_encoder(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			angle = decode_euler_2(decode_euler_1(e))
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(self.activation)(K_layer.add([decode_repete(angle), residual]))
			return angle

		angles = [None]*len(self.sup_hierarchies)
		for i,k in enumerate(self.sup_hierarchies):
			e = K_layer.Lambda(lambda x: x[:,k], output_shape=(self.latent_dim,))(encoded)
			angles[i] = decode_angle(e)

		decoded =  K_layer.concatenate(angles, axis=1)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)