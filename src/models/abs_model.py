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
		self.embedding = None

		self.latent_dim = args['latent_dim']
		self.input_dim = args['input_data_stats']['data_dim']

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

		if args['lstm']:
			from keras.layers import LSTM as RNN_UNIT
		else:
			from keras.layers import GRU as RNN_UNIT

		self.make_model()

	def load(self, load_path):
		self.model.load_weights(load_path)

	def autoencode(self, x):
		return self.model.predict(x)

	def encode(self, x):
		return self.encoder.predict(x)

	def decode(self, x):
		return self.decoder.predict(x)

	@abstractmethod
	def make_model(self): # build model
		pass

	@abstractmethod
	def load_embedding(self, data,**kwargs):
		'''
		Populate self.embedding
		'''
		pass

	@abstractmethod
	def format_data(self, x, **kwargs):
		'''
		Format the data to match the input and output of the model.
		'''
		pass

	@abstractmethod
	def predict(self, x, **kwargs):
		'''
		Default motion prediction algorithm
		Input motion sequence and predict the rest
		'''
		pass

	@abstractmethod
	def classify(self, x, **kwargs):
		'''
		Default motion classification algorithm
		Input motion sequence and infer the action name
		'''
		pass