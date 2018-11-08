import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter

class HH_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in'] # this is the number of frame we want to input for comparing against prediction baselines
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps, self.unit_t)

		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps_in-1 in self.hierarchies
		assert self.timesteps-1 == self.hierarchies[-1]

		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps/self.unit_t
		self.sup_hierarchies = [self._get_sup_index(h) for h in self.hierarchies]

		return super(HH_RNN, self).__init__(args)

	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		encode_1 = abs_model.RNN_UNIT(self.latent_dim/2)
		encode_2 = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			angle = decode_euler_2(decode_euler_1(e))
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(self.activation)(K_layer.add([decode_repete(angle), residual]))
			return angle

		angles = [None]*len(self.sup_hierarchies)
		for i,k in enumerate(self.sup_hierarchies):
			e = K_layer.Lambda(lambda x: x[:,k], output_shape=(self.latent_dim,))(encoded)
			angles[i] = decode_angle(e)

		decoded =  K_layer.concatenate(angles, axis=1)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

	def _get_sup_index(self, i):
		return (i+1)/self.unit_t-1

	def load_embedding(self, data, **kwargs):
		m, sets, data = embedding_utils.parse_load_embedding(self, data, **kwargs)

		if m == embedding_utils.TIME_MODALITIES:
			zs = self.encoder.predict(data)
			for i in sets:
				z_i = self._get_sup_index(i)
				if i not in self.embedding:
					self.embedding[i] = zs[:,z_i]
				else:
					self.embedding[i] = np.concatenate([self.embedding[i], zs[:,z_i]])
		else:
			for i in sets:
				zs = self.encoder.predict(data[i])
				if i not in self.embedding:
					self.embedding[i] = zs[:,-1]
				else:
					self.embedding[i] = np.concatenate([self.embedding[i], zs[:,-1]])


	def format_data(self, x, **kwargs):
		'''
		Reformat the output data for computing the autoencoding error
		Same as H_RNN
		'''
		return formatter.format_h_rnn(self, x, **kwargs)

	# override
	def encode(self, x, modality=-1):
		z = self.encoder.predict(x)
		if type(modality) == int and modality > 0:
			return z[:,self._get_sup_index(modality)]
		elif type(modality) == type([]):
			modality = [self._get_sup_index(m) for m in modality]
			return z[:, modality]
		return z

	def predict(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		kwargs['return_seq_fn'] = lambda x: x[:,self.timesteps_in:]
		return pattern_matching.raw_match(x, self, **kwargs)

	def classify(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# from motion modality to motion+name modality
		kwargs['partial_encode_idx'] = self.timesteps-1
		kwargs['modality_partial'] = self.embedding['motion']
		kwargs['modality_complete'] = self.embedding['both']
		kwargs['return_seq_fn'] = lambda x: x[:,:,-self.name_dim:]

		# default using ADD method for pattern matching
		return pattern_matching.raw_match(x, self, **kwargs)
