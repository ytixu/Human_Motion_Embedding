import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter

class Seq2SeqAE(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		return super(HH_RNN, self).__init__(args)

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repeat = RepeatVector(self.timesteps)
		decode_rnn = abs_model.RNN_UNIT(self.input_dim, return_sequences=True)

		decoded = decode_rnn(decode_repeat(encoded))
		decoded_ = decode_rnn(decode_repeat(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

	def load_embedding(self, data, **kwargs):
		return

	def format_data(self, x, **kwargs):
		return x, x

	# override
	def encode(self, x, modality=-1):
		return self.encoder.predict(x)

	def predict(self, x, **kwargs):
		return self.decoder.predict(x)

	def classify(self, x, **kwargs):
		return self.decoder.predict(x)
