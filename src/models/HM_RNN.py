# This is only for multimodal data

import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter

class HM_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		assert args['supervised']

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

		# for different sensorimotor modalities
		self.signal_dim = [args['input_data_stats']['data_dim'], len(args['actions'])]
		self.signal_types = formatter.EXPAND_NAMES_MODALITIES # motion, name, both
		self.signal_n = len(self.signal_types)

		self.sup_hierarchies = [self.__get_sup_index(h, s) for s in self.signal_types for h in self.hierarchies]

		return super(HM_RNN, self).__init__(args)


	def make_model(self):
		activation = 'tanh'

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)

		# different encoders for different modalities
		modal_encode = [None]*len(self.signal_dim)
		for i,d in enumerate(self.signal_dim):
			modal_encode[i] = abs_model.RNN_UNIT(self.latent_dim/2)
		# global encoder for all the modalities together
		global_encode = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)

		unit_encode_reshape = K_layer.Reshape((self.unit_n, self.latent_dim/2))
		global_encode_reshape = K_layer.Reshape((self.unit_n*self.signal_n, self.latent_dim))

		def encode_partials(seq, seq_dim, encoder):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, seq_dim))(seq)
				encoded[i] = encoder(rs)
			return unit_encode_reshape(K_layer.concatenate(encoded, axis=1))

		def encoder_modalities(seq):
			encoded = None
			# assume 2 modalities for now
			# TODO: fix this, not hard-code it
			idx = self.signal_dim
			idx = [(0,idx[0]), (idx[0], self.input_dim)]

			encoded = [None]*self.signal_n
			for i,k in enumerate(idx):
				s,e = k
				rs = K_layer.Lambda(lambda x: x[:,:,:,s:e], output_shape=(self.timesteps, self.latent_dim/2))(seq)
				encoded[i] = encode_partials(rs, self.signal_dim[i], modal_encode[i])
				encoded[i] = global_encode(encoded[i])

			encoded[2] = K_layer.add([encoded[0], encoded[1]])
			# for i in range(self.signal_n):
			# 	encoded[i] = global_encode(encoded[i])

			return global_encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encoder_modalities(reshaped)

		# same as in HH_RNN
		z = K_layer.Input(shape=(self.latent_dim,))
		decode_euler_1 = K_layer.Dense(self.latent_dim/2, activation=activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=activation)

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.latent_dim/2, return_sequences=True, activation=activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=activation)

		def decode_angle(e):
			angle = decode_euler_2(decode_euler_1(e))
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(activation)(K_layer.add([decode_repete(angle), residual]))
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

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

	def __get_sup_index(self, m_t=-1, m_s=''):
		if m_s == '':
			return self.__get_sup_t_idx(m_t)
		elif m_t == -1:
			return self.__get_sup_s_idx(m_s)

		m_s = self.signal_types.index(m_s)
		return self.unit_n*m_s + (m_t+1)/self.unit_t-1

	def __get_sup_t_idx(self, m_t):
		return [self.unit_n*i + (m_t+1)/self.unit_t-1 for i in range(3)]

	def __get_sup_s_idx(self, m_s):
		m_s = self.signal_types.index(m_s)
		return [self.unit_n*m_s + i for i in range(self.unit_n)]

	def load_embedding(self, data, **kwargs):
		m, sets, data = embedding_utils.parse_load_embedding(self, data, **kwargs)

		def add_zs(i, z_i, zs):
			zs = np.reshape(zs[:,z_i], (-1, self.latent_dim))
			if i not in self.embedding:
				self.embedding[i] = zs
			else:
				self.embedding[i] = np.concatenate([self.embedding[i], zs])

		if m == embedding_utils.TIME_MODALITIES:
			zs = self.encoder.predict(data)
			for i in sets:
				z_i = self.__get_sup_t_idx(i)
				add_zs(i, z_i, zs)
		else:
			for i in sets:
				zs = self.encoder.predict(data[i])
				z_i = self.__get_sup_s_idx(i)
				add_zs(i, z_i, zs)

	def format_data(self, x, **kwargs):
		'''
		Reformat the output data for computing the autoencoding error
		Same as H_RNN
		'''
		kwargs['expand_all_names'] = True
		return formatter.format_h_rnn(self, x, **kwargs)

	# override
	def encode(self, x, modality=(-1,'')):
		m_t,m_s = modality
		z = self.encoder.predict(x)
		if m_t > 0 or m_s != '':
			return z[:,self.__get_sup_index(m_t, m_s)]
		return z

	def predict(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None
		if 'partial_encode_idx' not in kwargs:
			kwargs['partial_encode_idx'] = (self.timesteps_in-1, 'both')
		kwargs['return_seq_fn'] = lambda x: x[:,self.timesteps_in:]

		return pattern_matching.raw_match(x, self, **kwargs)

	def classify(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# from motion modality to motion+name modality
		kwargs['partial_encode_idx'] = (self.timesteps-1, 'motion')
		kwargs['modality_partial'] = self.embedding['motion']
		kwargs['modality_complete'] = self.embedding['both']
		kwargs['return_seq_fn'] = lambda x: x[:,:,-self.name_dim:]

		# default using ADD method for pattern matching
		return pattern_matching.raw_match(x, self, **kwargs)
