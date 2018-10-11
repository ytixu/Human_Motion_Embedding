import numpy as np

import keras.layers as K_layer
from keras.models import Model

import abs_model
from embedding import pattern_matching, embedding_utils
from format_data import formatter

class VL_RNN(abs_model.AbstractModel):
	def __init__(self, args):
		self.timesteps = args['timesteps']
		self.timesteps_in = args['timesteps_in']
		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']
                self.hierarchies = map(int, args['hierarchies']) if args['hierarchies'] is not None else range(self.timesteps)

		# indices relevant to prediction task must appear in hierarchies
		assert(self.timesteps_in-1 in self.hierarchies)
		assert(self.timesteps-1 in self.hierarchies)

		self.output_dim = self.input_dim

		return super(VL_RNN, self).__init__(args)

	def make_model(self):
		# Same as seq2seq.py
		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		encoded = abs_model.RNN_UNIT(self.latent_dim)(inputs)

		z = K_layer.Input(shape=(self.latent_dim,))
		decoder_activation = 'tanh'
		decode_repete = K_layer.RepeatVector(self.timesteps)
		decode_rnn = abs_model.RNN_UNIT(self.output_dim, return_sequences=True, activation=decoder_activation)

		decoded = decode_rnn(decode_repete(encoded))
		decoded_ = decode_rnn(decode_repete(z))

		self.encoder = Model(inputs, encoded)
		self.decoder = Model(z, decoded_)
		self.model = Model(inputs, decoded)

		self.model.compile(optimizer=self.opt, loss=self.loss_func)

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
		if 'for_validation' in kwargs and kwargs['for_validation']:
			return x, x
		if self.supervised:
			x = formatter.randomize_name(self, x)
		return formatter.expand_time_vl(self, x)

	# override
	def encode(self, x, modality=-1):
		return self.encoder.predict(x)

	def predict(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# default using ADD method for pattern matching
		return pattern_matching.raw_match(x, self, **kwargs)

	def classify(self, x, **kwargs):
		# assume data is alrady formatted
		# and embedding is loaded
		assert self.embedding != None

		# from motion modality to motion+name modality
		kwargs['partial_encode_idx'] = model.timesteps-1
		kwargs['modality_partial'] = 'motion'
		kwargs['modality_complete'] = 'both'

		# default using ADD method for pattern matching
		return pattern_matching.raw_match(x, self, **kwargs)