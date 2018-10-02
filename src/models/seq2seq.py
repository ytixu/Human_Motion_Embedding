import json

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
		encoded = RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_residual = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		decoded = decode_residual(decode_repete(encoded))
		decoded_ = decode_residual(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

		self.model.summary()
		self.encoder.summary()
		self.decoder.summary()

	def load(self, load_path):
		self.model.load_weights(load_path)

	def format_data(self, x):
		return x[:,:self.timesteps_in], x[:,self.timesteps_in:]

	def predict(self, x):
		return self.model.predict(x)
