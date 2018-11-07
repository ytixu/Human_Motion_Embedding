import numpy as np

import keras.layers as K_layer
import keras.backend as K_backend
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter
import HH_RNN

class SH_RNN(HH_RNN.HH_RNN):
	# seperate name from motion
	def __init__(self, args):
		assert args['supervised']

		return super(SH_RNN, self).__init__(args)

	def make_model(self):
		self.partial_latent_dim = self.latent_dim

		# encoder
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		name_slice = K_layer.Lambda(lambda x: x[:,:,:,-self.name_dim:], output_shape=(self.unit_n, self.unit_t, self.name_dim))(reshaped)
		motion_slice = K_layer.Lambda(lambda x: x[:,:,:,:-self.name_dim], output_shape=(self.unit_n, self.unit_t, self.input_dim-self.name_dim))(reshaped)

		encode_name = abs_model.RNN_UNIT(self.partial_latent_dim)
		encode_motion = abs_model.RNN_UNIT(self.partial_latent_dim)
		encode_reshape = K_layer.Reshape((self.unit_n, self.partial_latent_dim))
		encode_global = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)

		def encode_partials(seq, encoder, shape):
			partial = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, shape))(seq)
				partial[i] = encoder(rs)
			partial = encode_reshape(K_layer.concatenate(partial, axis=1))
			return encode_global(partial)

		encoded_name = encode_partials(name_slice, encode_name, self.name_dim)
		encoded_motion = encode_partials(motion_slice, encode_motion, self.input_dim-self.name_dim)
		encoded = K_layer.add([encoded_name, encoded_motion])
		#encoded = encode_global(encoded)

		# decoder
		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repeat_units = K_layer.RepeatVector(self.unit_n)
		decode_units = abs_model.RNN_UNIT(self.partial_latent_dim, return_sequences=True, activation=self.activation)

		#decode_name_1 = K_layer.Dense(128, activation=self.activation)
		decode_name_2 = K_layer.Dense(self.name_dim, activation=self.activation)
		#decode_motion_1 = K_layer.Dense(236, activation=self.activation)
		decode_motion_2 = K_layer.Dense(self.output_dim-self.name_dim, activation=self.activation)
		decode_repete_angles = K_layer.Lambda(lambda x:K_backend.repeat_elements(x, self.unit_t, 1), output_shape=(self.timesteps, self.output_dim))

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(512, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			units = decode_units(decode_repeat_units(e))
			#name = K_layer.TimeDistributed(decode_name_1)(units)
			name = K_layer.TimeDistributed(decode_name_2)(units)
			motion = K_layer.TimeDistributed(decode_motion_2)(units)
			#motion = K_layer.TimeDistributed(decode_motion_2)(motion)
			seq = decode_repete_angles(K_layer.concatenate([motion, name], axis=-1))

			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			seq = K_layer.Activation(self.activation)(K_layer.add([seq, residual]))
			return seq

		angles = [None]*len(self.sup_hierarchies)
		for i,k in enumerate(self.sup_hierarchies):
			e = K_layer.Lambda(lambda x: x[:,k], output_shape=(self.latent_dim,))(encoded)
			angles[i] = decode_angle(e)

		decoded =  K_layer.concatenate(angles, axis=1)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)
