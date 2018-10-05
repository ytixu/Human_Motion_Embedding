import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import optimizers

RNN_UNIT = None

class Seq2Seq:
	def __init__(self, args):
		global RNN_UNIT

		self.model = None
		self.encoder = None
		self.decoder = None

		self.timesteps_out = args['timesteps_out']
		self.timesteps_in = args['timesteps_in']
		self.hierarchies = args['hierarchies']
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
		inputs = K_layer.Input(shape=(self.timesteps_in, self.input_dim))
		encoded = RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_rnn = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

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

		self.model.summary()
		self.encoder.summary()
		self.decoder.summary()

	def load(self, load_path):
		self.model.load_weights(load_path)

	def load_embedding(self, data, pred_only=False):
		# assume data is alrady formatted
		if self.embedding is None:
			self.embedding = [[]]*self.timesteps

		sets = [self.timesteps_in-1, t-1] if pred_only else self.hierarchies

		zs = self.encoder.predict(data[i])
		for i in sets:
			if len(self.embedding[i]) == 0:
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

	def autoencode(self, x):
		return self.model.predict(x)

	def predict(self, x, return_std=False):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		c = self.timesteps_in-1
		z_ref = self.encoder.predict(x)[:,c]
		# TODO: add other methods
		z_pred = pattern_matching.add(self.embedding[c], self.embedding[-1], z_ref, return_std=return_std)

		if return_std:
			std, z_pred = z_pred
			return std, self.decoder.predict(z_pred)

		return self.decoder.predict(z_pred)
