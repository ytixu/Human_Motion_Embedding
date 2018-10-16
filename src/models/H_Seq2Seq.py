import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from utils import pattern_matching

class H_Seq2Seq(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps_in = args['timesteps_in']
		self.timesteps_out = args['timesteps_out']
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps_in, self.unit_t)
		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps_in-1 == self.hierarchies[-1]
		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps_in/self.unit_t
		self.sup_hierarchies = [self.__get_sup_index(h) for h in self.hierarchies]

		return super(H_Seq2Seq, self).__init__(args)

	def make_model(self):
		# Similar to HH_RNN
		inputs = K_layer.Input(shape=(self.timesteps_in, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		encode_1 = abs_model.RNN_UNIT(self.latent_dim/2)
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
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)

		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_residual_1 = abs_model.RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		decoded =  decode_residual_2(decode_residual_1(decode_repete(encoded)))
		decoded_ = decode_residual_2(decode_residual_1(decode_repete(z)))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def __get_sup_index(self, i):
		return (i+1)/self.unit_t-1

	def load_embedding(self, data, pred_only=False, new=False):
		pass

	def format_data(self, x, **kwargs):
		# Same as Seq2Seq
		return x[:,:self.timesteps_in], x[:,self.timesteps_in:]

	def predict(self, x, **kwargs):
		# Same as Seq2Seq
		return self.model.predict(x)

	def classify(self, x, **kwargs):
		# not supported
		return None
