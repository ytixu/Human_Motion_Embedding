import numpy as np

import keras.layers as K_layer
import keras.backend as K_backend
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter
import H_Seq2Seq

class HH_Seq2Seq_PC(H_Seq2Seq.H_Seq2Seq):

	def make_model(self):
		self.partial_latent_dim = self.latent_dim/2
		self.unit_n_out = self.timesteps/self.unit_t

		inputs = K_layer.Input(shape=(self.timesteps_in, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.partial_latent_dim))
		encode_1 = abs_model.RNN_UNIT(self.partial_latent_dim)
		encode_2 = abs_model.RNN_UNIT(self.latent_dim)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repeat_units = K_layer.RepeatVector(self.unit_n_out)
		decode_units = abs_model.RNN_UNIT(self.partial_latent_dim, return_sequences=True, activation=self.activation)

		decode_euler_1 = K_layer.Dense(self.output_dim*4, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)
		decode_repete_angles = K_layer.Lambda(lambda x:K_backend.repeat_elements(x, self.unit_t, 1), output_shape=(self.timesteps, self.output_dim))

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.output_dim*4, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			angle = decode_units(decode_repeat_units(e))
			angle = K_layer.TimeDistributed(decode_euler_1)(angle)
			angle = K_layer.TimeDistributed(decode_euler_2)(angle)
			angle = decode_repete_angles(angle)
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(self.activation)(K_layer.add([angle, residual]))
			return angle

		decoded = decode_angle(encoded)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)


	def format_data(self, x, **kwargs):
		return x[:,:self.timesteps_in], x

	def predict(self, x, **kwargs):
		return self.model.predict(x)[:,self.timesteps_in:]
