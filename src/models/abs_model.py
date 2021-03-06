'''
Abstract class for models
'''

from abc import ABCMeta, abstractmethod
import numpy as np

from keras import optimizers
from utils import pattern_matching

RNN_UNIT = None

class AbstractModel:
	__metaclass__ = ABCMeta

	def __init__(self, args):
		global RNN_UNIT

		self.model = None
		self.encoder = None
		self.decoder = None
		self.embedding = {}

		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']

		if args['normalization_method'] in ['norm_pi', 'norm_max']:
			self.activation = 'tanh'
		else:
			self.activation = 'linear'

		self.add_noise = False
		if args['add_noise']:
			self.add_noise = True
			self.noise_std = args['noise_std']

		self.repeat_last = args['repeat_last']
		self.supervised = args['supervised']

		if self.supervised:
			self.name_dim = len(args['actions'])
			self.actions = args['actions']
			self.input_dim += self.name_dim

		# TODO: different output
		self.output_dim = self.input_dim

		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])
		self.lr = args['learning_rate']
		self.decay = args['decay']

		if args['lstm']:
			from keras.layers import LSTM as RNN_UNIT
		else:
			from keras.layers import GRU as RNN_UNIT

		self.make_model()
		self.model.compile(optimizer=self.opt(lr=self.lr), loss=self.loss_func)

	def load(self, load_path):
		self.model.load_weights(load_path)

	def decay_learning_rate(self):
		if self.decay < 1:
			self.lr = self.lr * self.decay
			self.model.compile(optimizer=self.opt(lr=self.lr), loss=self.loss_func)

	def autoencode(self, x):
		return self.model.predict(x)

	def encode(self, x):
		return self.encoder.predict(x)

	def decode(self, x):
		return self.decoder.predict(x)

	@abstractmethod
	def make_model(self): # build model
		pass

	def reset_embedding(self):
		'''
		Reset embedding
		'''
		self.embedding = {}

	@abstractmethod
	def load_embedding(self, data,**kwargs):
		'''
		Populate self.embedding
		'''
		pass

	@abstractmethod
	def format_data(self, x, **kwargs):
		'''
		Format the data to match the input and output of the model,
		according the task (prediction, classification)
		'''
		pass

	@abstractmethod
	def predict(self, x, **kwargs):
		'''
		Default motion prediction algorithm
		Given formatted x_1:t for prediction,
		Output x_t+1:T
		'''
		pass

	@abstractmethod
	def classify(self, x, **kwargs):
		'''
		Default motion classification algorithm
		Given formatted x_1:T for classification,
		Output the action name l_1:T
		'''
		pass
