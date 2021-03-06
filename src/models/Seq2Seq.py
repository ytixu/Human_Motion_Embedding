import numpy as np

import keras.layers as K_layer
from keras.models import Model
import keras.backend as K_backend

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
		#decode_repete = K_layer.RepeatVector(1)
		#decode_rnn = abs_model.RNN_UNIT(self.latent_dim, return_state=True, return_sequences=True, activation=self.activation)
		decode_repete = K_layer.RepeatVector(self.timesteps_out)
		decode_rnn = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		dense = K_layer.TimeDistributed(K_layer.Dense(self.output_dim))
		last_frame = K_layer.Lambda(lambda x: x[:,-1], output_shape=(self.output_dim,))(inputs)


		# no teacher forcing
		def decode(inputs):
			all_outputs = [None]*self.timesteps_out
			states = None
			for i in range(self.timesteps_out):
				if states is None:
					inputs = decode_repete(inputs)
					outputs, states = decode_rnn(inputs)
				else:
					outputs, states = decode_rnn(inputs, initial_state=states)
				all_outputs[i] = outputs
				inputs = outputs
			outputs = K_layer.Lambda(lambda x: K_layer.concatenate(x, axis=1))(all_outputs)
			return dense(outputs)

		#decoded = decode(encoded)
		#decoded_ = decode(z)

		# residual
		#decode_rnn_2 = abs_model.RNN_UNIT(self.output_dim*4, return_sequences=True, activation=self.activation)


		decoded = decode_rnn(decode_repete(encoded))
		decoded_ = decode_rnn(decode_repete(z))

		#	last_frame = K_layer.Dense(self.output_dim*4)(encoded)
		#last_frame = K_layer.Dense(self.output_dim)(encoded)
		#last_frame_ = K_layer.Dense(self.output_dim)(z)

		#decoded = K_layer.add([decode_repete(last_frame), decoded])
		#decoded_ = K_layer.add([decode_repete(last_frame_), decoded_])

		outputs = [None]*self.timesteps_out
		first_decoded = K_layer.Lambda(lambda x: x[:,0], output_shape=(self.output_dim,))(decoded)
		outputs[0] = K_layer.add([first_decoded,last_frame])
		for i in range(1, self.timesteps_out):
			dec_ = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.output_dim,))(decoded)
			outputs[i] = K_layer.add([dec_,outputs[i-1]])
		decoded =  K_layer.concatenate(outputs, axis=1)
		decoded = K_layer.Reshape((self.timesteps_out, self.output_dim))(decoded)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

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
