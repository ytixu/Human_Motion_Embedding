import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import backend as K

import abs_model

class C_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		assert args['supervised']

		self.timesteps = args['timesteps']

		#########
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps, self.unit_t)
		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps-1 == self.hierarchies[-1]
		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps/self.unit_t
		self.sup_hierarchies = [self.__get_sup_index(h) for h in self.hierarchies]
		###########

		return super(C_RNN, self).__init__(args)

	def make_model(self):
		self.input_dim = self.input_dim - self.name_dim
                self.output_dim = self.name_dim

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		encode_1 = abs_model.RNN_UNIT(self.latent_dim/2)
		encode_2 = abs_model.RNN_UNIT(self.latent_dim)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		decoded = K_layer.Dense(self.output_dim, activation='sigmoid')(encoded)
		decoded = K_layer.Lambda(lambda x: K.tf.nn.softmax(x))(decoded)
		output = K_layer.RepeatVector(self.timesteps)(decoded)
		self.model = Model(inputs, output)
                self.encoder = self.model
                self.decoder = self.model

	def __get_sup_index(self, i):
		return (i+1)/self.unit_t-1

	def back_make_model(self):
		self.input_dim = self.input_dim - self.name_dim
		self.output_dim = self.name_dim

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		#encoded = K_layer.Lambda(lambda x: x[:,-1], output_shape=(self.input_dim,))(inputs)
		#encoded = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)
		encoded = K_layer.Flatten()(inputs)
		#decoded = K_layer.Dense(self.latent_dim, activation='relu')(decoded)
		#decoded = K_layer.Dense(self.latent_dim/2, activation='relu')(decoded)
		decoded = K_layer.Dense(self.output_dim, activation='sigmoid')(encoded)
		decoded = K_layer.Lambda(lambda x: K.tf.nn.softmax(x))(decoded)
		output = K_layer.RepeatVector(self.timesteps)(decoded)

		self.model = Model(inputs, output)
		self.encoder = self.model
		self.decoder = self.model

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

