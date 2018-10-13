import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model

class Seq2Seq(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps_out = args['timesteps_out']
		self.timesteps_in = args['timesteps_in']

		return super(Seq2Seq, self).__init__(args)

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps_in, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_rnn = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		decoded = decode_rnn(decode_repete(encoded))
		decoded_ = decode_rnn(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def load_embedding(self, data, **kwargs):
		# no embedding
		pass

	def format_data(self, x, **kwargs):
		return x[:,:self.timesteps_in], x[:,self.timesteps_in:]

	def predict(self, x, **kwargs):
		return self.model.predict(x)

	def classify(self, x, **kwargs):
		# not supported
		return None
