import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import optimizers

from embedding import pattern_matching

RNN_UNIT = None

class VL_RNN:
	def __init__(self, args):
		global RNN_UNIT

		self.model = None
		self.encoder = None
		self.decoder = None
		self.embedding = None

		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in']
		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']
		self.output_dim = self.input_dim

		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])

		self.repeat_last = args['repeat_last']

		if args['lstm']:
			from keras.layers import LSTM as RNN_UNIT
		else:
			from keras.layers import GRU as RNN_UNIT

		self.make_model()


	def make_model(self):
		# Same as seq2seq.py
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_rnn = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		decoded = decode_rnn(decode_repete(encoded))
		decoded_ = decode_rnn(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def load(self, load_path):
		self.model.load_weights(load_path)

	def load_embedding(self, data, pred_only=False):
		# assume data is alrady formatted
		n,t,d = data.shape
		data = np.reshape(data, (t,n/t,t,d))

		if self.embedding is None:
			self.embedding = [[]]*self.timesteps

		sets = [self.timesteps_in-1, t-1] if pred_only else range(t)

		for i in sets:
			zs = self.encoder.predict(data[i])
			if len(self.embedding[i]) == 0:
				self.embedding[i] = zs
			else:
				self.embedding[i] = np.concatenate([self.embedding[i], zs])


	def format_data(self, x):
		'''
		Reformat the data so that we can encode sequences of different lengths.
		'''
		n,t,d = x.shape
		new_x = np.zeros((n*self.timesteps, t, d))

		for i in range(self.timesteps):
			# the first i frames
			new_x[i*n:(i+1)*n,:i+1] = x[:,:i+1]
			# the rest of the frames
			if self.repeat_last and i+1 != self.timesteps:
				for j in range(i*n,(i+1)*n):
					new_x[j,i+1:] = x[j,i]
		return new_x, new_x

	def autoencode(self, x):
		return self.model.predict(x)

	def predict(self, x, method=pattern_matching.ADD, return_std=False):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		n,t,d = x.shape
		x = np.reshape(x, (t,n/t,t,d))

		c = self.timesteps_in-1
		z_ref = self.encoder.predict(x[c])
		# TODO: add other methods
		z_pred = pattern_matching.add(self.embedding[c], self.embedding[-1], z_ref, return_std=return_std)

		if return_std:
			std, z_pred = z_pred
			return std, self.decoder.predict(z_pred)

		return self.decoder.predict(z_pred)
