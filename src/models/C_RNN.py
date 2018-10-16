import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import backend as K

import abs_model

class C_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		assert args['supervised']

		self.timesteps = args['timesteps']
		return super(C_RNN, self).__init__(args)

	def make_model(self):
		self.input_dim = self.input_dim - self.name_dim
                self.output_dim = self.name_dim

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)
		decoded = abs_model.RNN_UNIT(self.output_dim)(encoded)
		decoded = K_layer.Lambda(lambda x: K.tf.nn.softmax(x))(decoded)
		output = K_layer.RepeatVector(self.timesteps)(decoded)

		self.model = Model(inputs, output)
		self.encoder = self.model
		self.decoder = self.model
		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def load_embedding(self, data, **kwargs):
		# no embedding
		pass

	def format_data(self, x, **kwargs):
		return x[:,:,:-self.name_dim], x[:,:,-self.name_dim:]

	def predict(self, x, **kwargs):
		# not supported
		pass

	def classify(self, x, **kwargs):
		return self.model.predict(x)
