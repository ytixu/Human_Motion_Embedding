import numpy as np

import keras.layers as K_layer
import keras.backend as K_backend
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter
import HH_RNN

class SH_RNN(HH_RNN.HH_RNN):
	# 3 layers
	def __init__(self, args):
		assert args['supervised']

		self.sub_unit = 5
		self.partial_latent_dim = args['latent_dim']/2
		self.sub_partial_latent_dim = args['latent_dim']/4

		return super(SH_RNN, self).__init__(args)

	def make_model(self):
		assert self.sub_unit**2 = self.unit_t

		# ENCODER
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)

		reshaped_to_sub_units = K_layer.Reshape((self.sub_unit, self.sub_unit, self.input_dim))
		encode_reshape_1 = K_layer.Reshape((self.sub_unit, self.sub_partial_latent_dim))
		encode_reshape_2 = K_layer.Reshape((self.unit_n, self.partial_latent_dim))
		encode_1 = abs_model.RNN_UNIT(self.sub_partial_latent_dim)
		encode_2 = abs_model.RNN_UNIT(self.partial_latent_dim)
		encode_3 = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)

		def encode_partials(seq, encoder_layer, reshape_layer_1, reshape_layer_2=None):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				if reshape_layer is not None:
					rs = reshape_layer(rs)
				else:
					rs = encode_partials(rs, encode_1, encode_reshape_1, reshaped_to_sub_units)
				encoded[i] = encoder_layer(rs)
			return reshape_layer_1(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped, encode_2, encode_reshape_2)
		encoded = encode_3(encoded)


		# DECODER
		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repeat_units = K_layer.RepeatVector(self.unit_n)
		decode_repeat_elements = K_layer.Lambda(lambda x: K_backend.repeat_elements(x, rep=self.sub_unit, axis=1), output_shape=(self.unit_n*self.sub_unit, self.partial_latent_dim))
		decode_units_1 = abs_model.RNN_UNIT(self.partial_latent_dim, return_sequences=True, activation=self.activation)
		decode_units_2 = abs_model.RNN_UNIT(self.sub_partial_latent_dim, return_sequences=True, activation=self.activation)

		decode_euler_1 = K_layer.Dense(self.output_dim*4, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)
		decode_repete_dense = K_layer.Lambda(lambda x:K_backend.repeat_elements(x, self.sub_unit, 1), output_shape=(self.timesteps, self.output_dim))

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.output_dim*4, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			res = decode_units_1(decode_repeat_units(e))
			res = decode_repeat_elements(res)
			res = K_layer.TimeDistributed(decode_euler_1)(res)
			res = K_layer.TimeDistributed(decode_euler_2)(res)
			res = decode_repete_dense(res)
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			res = K_layer.Activation(self.activation)(K_layer.add([res, residual]))
			return res

		angles = [None]*len(self.sup_hierarchies)
		for i,k in enumerate(self.sup_hierarchies):
			e = K_layer.Lambda(lambda x: x[:,k], output_shape=(self.latent_dim,))(encoded)
			angles[i] = decode_angle(e)

		decoded =  K_layer.concatenate(angles, axis=1)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)
