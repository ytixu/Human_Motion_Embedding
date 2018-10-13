import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model

class C_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		assert args['supervised']

		self.timesteps = args['timesteps']
		return super(Seq2Seq, self).__init__(args)
		self.input_dim = args['data_dim']
		self.output_dim = args['data_dim']

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		output = abs_model.RNN_UNIT(self.latent_dim, activation="softmax")(inputs)

		self.model = Model(inputs, output)
		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def load_embedding(self, data, pred_only=False, new=False):
		# no embedding
		pass

	def format_data(self, x, **kwargs):
		return x[:,:,:-self.name_dim], x[:,:,-self.name_dim:]

	def predict(self, x, **kwargs):
		# not supported
		pass

	def classify(self, x, **kwargs):
		return self.model.predict(x)