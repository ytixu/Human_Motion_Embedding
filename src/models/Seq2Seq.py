import numpy as np

import keras.layers as K_layer
from keras.models import Model
#from keras import optimizers

import abs_model

class Seq2Seq(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps_out = args['timesteps_out']
		self.timesteps_in = args['timesteps_in']
		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']
		self.output_dim = self.input_dim

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

	def load_embedding(self, data, pred_only=False, new=False):
		# no embedding
		pass

	def format_data(self, x):
		return x[:,:self.timesteps_in], x[:,self.timesteps_in:]

	def predict(self, x, return_std=False):
		x_pred = np.concatenate([x,self.model.predict(x)], axis=1)
		if return_std:
			return [], x_pred
		return x_pred
