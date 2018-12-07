import numpy as np

import keras.layers as K_layer
import keras.backend as K_backend
from keras.models import Model

import abs_model
from utils import pattern_matching, embedding_utils, formatter

class VL_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in']
		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']

		self.timesteps_out = args['timesteps_out']
		self.unit_t = args['unit_timesteps']
		self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.unit_t-1, self.timesteps, self.unit_t)
		# indices relevant to prediction task must appear in hierarchies
		assert self.timesteps_in-1 in self.hierarchies
		# hierarchies must be multiple of unit_t
		assert not any([(h+1)%self.unit_t for h in self.hierarchies])
		self.unit_n = self.timesteps/self.unit_t
		self.partial_latent_dim = 514

		#self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.timesteps)

		# indices relevant to prediction task must appear in hierarchies
		#assert(self.timesteps_in-1 in self.hierarchies)
		#assert(self.timesteps-1 in self.hierarchies)

		self.output_dim = self.input_dim

		return super(VL_RNN, self).__init__(args)


	def make_model(self):
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		reshaped = K_layer.Reshape((self.unit_n, self.unit_t, self.input_dim))(inputs)
		encode_reshape = K_layer.Reshape((self.unit_n, self.partial_latent_dim))
		encode_1 = abs_model.RNN_UNIT(self.partial_latent_dim)
		encode_2 = abs_model.RNN_UNIT(self.latent_dim)

		def encode_partials(seq):
			encoded = [None]*self.unit_n
			for i in range(self.unit_n):
				rs = K_layer.Lambda(lambda x: x[:,i], output_shape=(self.unit_t, self.input_dim))(seq)
				encoded[i] = encode_1(rs)
			return encode_reshape(K_layer.concatenate(encoded, axis=1))

		encoded = encode_partials(reshaped)
		encoded = encode_2(encoded)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repeat_units = K_layer.RepeatVector(self.unit_n)
		decode_units = abs_model.RNN_UNIT(self.partial_latent_dim, return_sequences=True, activation=self.activation)

		decode_euler_1 = K_layer.Dense(self.output_dim*4, activation=self.activation)
		decode_euler_2 = K_layer.Dense(self.output_dim, activation=self.activation)
		decode_repete_angles = K_layer.Lambda(lambda x:K_backend.repeat_elements(x, self.unit_t, 1), output_shape=(self.timesteps, self.output_dim))

		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_residual_1 = abs_model.RNN_UNIT(self.output_dim*4, return_sequences=True, activation=self.activation)
		decode_residual_2 = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		def decode_angle(e):
			angle = decode_units(decode_repeat_units(e))
			angle = K_layer.TimeDistributed(decode_euler_1)(angle)
			angle = K_layer.TimeDistributed(decode_euler_2)(angle)
			angle = decode_repete_angles(angle)
			residual = decode_repete(e)
			residual = decode_residual_2(decode_residual_1(residual))
			angle = K_layer.Activation(self.activation)(K_layer.add([angle, residual]))
			return angle

		decoded = decode_angle(encoded)
		decoded_ = decode_angle(z)

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

	def back_make_model(self):
		# Same as seq2seq.py
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_rnn = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=self.activation)

		decoded = decode_rnn(decode_repete(encoded))
		decoded_ = decode_rnn(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

	def load_embedding(self, data, **kwargs):
		m, sets, data = embedding_utils.parse_load_embedding(self, data, **kwargs)

		if m == embedding_utils.TIME_MODALITIES:
			#reformat data to be key-ed by the modality
			n,k,d = data.shape
			t = len(self.hierarchies)
			data = np.reshape(data, (t,n/t,k,d))
			data = {h:data[i] for i,h in enumerate(self.hierarchies)}

		# populate
		for i in sets:
			zs = self.encoder.predict(data[i])
			if i not in self.embedding:
				self.embedding[i] = zs
			else:
				self.embedding[i] = np.concatenate([self.embedding[i], zs])

	def format_data(self, x, **kwargs):
		'''
		Reformat the data so that we can encode sequences of different lengths.
		'''
		# TODO: fix this
		if 'for_prediction' in kwargs and kwargs['for_prediction']:
			return x, x[:,self.timesteps_in:]
		if self.supervised:
			x = formatter.randomize_name(self, x)
		return formatter.expand_time_vl(self, x)

	# override
	def encode(self, x, modality=-1):
		if type(modality) == int and (modality > 0 or modality < self.timesteps-1):
			modality  = [modality]
		if type(modality) == type([]):
			new_x = np.zeros((x.shape[0], len(modality), self.timesteps, x.shape[-1]))
			for i,m in enumerate(modality):
				new_x[:,i] = x
				for j in range(m,self.timesteps):
					new_x[:,i,j] = x[:,m]
			new_x = new_x.reshape((-1, self.timesteps, x.shape[-1]))
			z = self.encoder.predict(new_x)
			if len(modality) > 1:
				return z.reshape((x.shape[0],len(modality),-1))
			return z

		return self.encoder.predict(x)

	def predict(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# default using ADD method for pattern matching
		kwargs['return_seq_fn'] = lambda x: x[:,self.timesteps_in:]
		return pattern_matching.raw_match(x, self, **kwargs)

	def classify(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# from motion modality to motion+name modality
		kwargs['partial_encode_idx'] = self.timesteps-1
		kwargs['modality_partial'] = 'motion'
		kwargs['modality_complete'] = 'both'
		kwargs['return_seq_fn'] = lambda x: x[:,:,-self.name_dim:]

		# default using ADD method for pattern matching
		return pattern_matching.raw_match(x, self, **kwargs)
