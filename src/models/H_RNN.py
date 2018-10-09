import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from embedding import pattern_matching
from format_data import formatter

class H_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in'] # this is the number of frame we want to input for comparing against prediction baselines
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

	def load_embedding(self, data, **kwargs):
		embedding_utils.load_embedding(model, data, **kwargs)

	def format_data(self, x, **kwargs):
		'''
		Reformat the output data for computing the autoencoding error
		Same as HH_RNN
		'''
		return formatter.expand_modalities(self, x, **kwargs)

	# overrides
	def encode(self, x, modality=-1):
		z = self.encoder.predict(x)
		if modality > 0:
			return z[:,modality]

	def predict(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		return pattern_matching.raw_match(x, self, **kwargs)
