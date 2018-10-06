import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import optimizers

from embedding import pattern_matching

RNN_UNIT = None

class HH_RNN:
	def __init__(self, args):
		global RNN_UNIT

		self.model = None
		self.encoder = None
		self.decoder = None
		self.embedding = None

		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in']
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps, self.unit_t)
		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps_in-1 in self.hierarchies
		assert self.timesteps-1 == self.hierarchies[-1]
		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps/self.unit_t
		self.sup_hierarchies = [self.__get_sup_index(h) for h in self.hierarchies]

		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']
		self.output_dim = self.input_dim

		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])

		if args['lstm']:
			from keras.layers import LSTM as RNN_UNIT
		else:
			from keras.layers import GRU as RNN_UNIT

		self.make_model()

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		encode_1 = RNN_UNIT(self.latent_dim/2)
		encode_2 = RNN_UNIT(self.latent_dim, return_sequences=True)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=decoder_activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=decoder_activation)

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=decoder_activation)
		decode_residual_2 = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		def decode_angle(e):
			angle = decode_euler_2(decode_euler_1(e))
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(decoder_activation)(K_layer.add([decode_repete(angle), residual]))
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

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def __get_sup_index(self, i):
		return (i+1)/self.unit_t-1

	def load(self, load_path):
		self.model.load_weights(load_path)

	def load_embedding(self, data, pred_only=False, new=False):
		# assume data is alrady formatted
		if new or self.embedding is None:
			self.embedding = {}

		sets = [self.timesteps_in-1, t-1] if pred_only else self.hierarchies

		zs = self.encoder.predict(data)
		for i in sets:
			z_i = self.__get_sup_index(i)
			if i not in embedding:
				self.embedding[i] = zs[:,z_i]
			else:
				self.embedding[i] = np.concatenate([self.embedding[i], zs[:,z_i]])

	def format_data(self, x):
		'''
		Reformat the output data for computing the autoencoding error
		'''
		y = np.repeat(x, len(self.hierarchies), axis=0)
		y = np.reshape(y, (-1, len(self.hierarchies), self.timesteps, y.shape[-1]))
		for i, h in enumerate(self.hierarchies):
			for j in range(h+1, self.timesteps):
				y[:,i,j] = y[:,i,h]
		y = np.reshape(y, (-1, self.timesteps*len(self.hierarchies), y.shape[-1]))
		return x, y

	def autoencode(self, x):
		return self.model.predict(x)

	def predict(self, x, return_std=False):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		c = self.timesteps_in-1
		z_ref = self.encoder.predict(x)[:,self.__get_sup_index(c)]
		# TODO: add other methods
		z_pred = pattern_matching.add(self.embedding[c], self.embedding[-1], z_ref, return_std=return_std)

		if return_std:
			std, z_pred = z_pred
			return std, self.decoder.predict(z_pred)

		return self.decoder.predict(z_pred)