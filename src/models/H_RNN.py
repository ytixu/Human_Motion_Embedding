import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from embedding import pattern_matching

class H_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.timesteps)
		# indices relevant to prediction task must appear in hierarchies
		assert(self.timesteps_in-1 in self.hierarchies)
		assert(self.timesteps-1 in self.hierarchies)

		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']
		self.output_dim = self.input_dim

		return super(H_RNN, self).__init__(args)

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_rnn = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		partials = [None]*len(self.hierarchies)
		for i,h in enumerate(self.hierarchies):
			e = K_layer.Lambda(lambda x: x[:,h], output_shape=(self.latent_dim,))(encoded)
			partials[i] = decode_rnn(decode_repete(e))

		decoded =  K_layer.concatenate(partials, axis=1)
		decoded_ = decode_rnn(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def load_embedding(self, data, pred_only=False, new=True):
		# assume data is alrady formatted
		if new or self.embedding is None:
			self.embedding = {}

		sets = [self.timesteps_in-1, t-1] if pred_only else self.hierarchies

		zs = self.encoder.predict(data)
		for i in sets:
			if i not in self.embedding:
				self.embedding[i] = zs[:,i]
			else:
				self.embedding[i] = np.concatenate([self.embedding[i], zs[:,i]])

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

	def predict(self, x, return_std=False):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		c = self.timesteps_in-1
		z_ref = self.encoder.predict(x)[:,c]
		# TODO: add other methods
		z_pred = pattern_matching.add(self.embedding[c], self.embedding[self.timesteps-1], z_ref, return_std=return_std)

		if return_std:
			std, z_pred = z_pred
			return std, self.decoder.predict(z_pred)

		return self.decoder.predict(z_pred)
