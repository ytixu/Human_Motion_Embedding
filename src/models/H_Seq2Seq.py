import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import optimizers

from embedding import pattern_matching

RNN_UNIT = None

class H_Seq2Seq:
	def __init__(self, args):
		global RNN_UNIT

		self.model = None
		self.encoder = None
		self.decoder = None
		self.embedding = None

		self.timesteps_in = args['timesteps_in']
		self.timesteps_out = args['timesteps_out']
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps, self.unit_t)
		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps_in-1 == self.hierarchies[-1]
		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps_in/self.unit_t
		self.sup_hierarchies = [self.__get_sup_index(h) for h in self.hierarchies]

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
		# Similar to HH_RNN
		inputs = K_layer.Input(shape=(self.timesteps_in, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		encode_1 = RNN_UNIT(self.latent_dim/2)
		encode_2 = RNN_UNIT(self.latent_dim)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=decoder_activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=decoder_activation)

		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_residual_1 = RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=decoder_activation)
		decode_residual_2 = RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		decoded =  decode_residual_2(decode_residual_1(decode_repete(e)))
		decoded_ = decode_residual_2(decode_residual_1(decode_repete(z)))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def __get_sup_index(self, i):
		return (i+1)/self.unit_t-1

	def load(self, load_path):
		self.model.load_weights(load_path)

	def load_embedding(self, data, pred_only=False, new=False):
		pass

	def format_data(self, x):
		# Same as Seq2Seq
		return x[:,:self.timesteps_in], x[:,self.timesteps_in:]

	def autoencode(self, x):
		return self.model.predict(x)

	def predict(self, x, return_std=False):
		# Same as Seq2Seq
		x_pred = np.concatenate([x,self.model.predict(x)], axis=1)
		if return_std:
			return [], x_pred
		return x_pred