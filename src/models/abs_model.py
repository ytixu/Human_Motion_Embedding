'''
Abstract class for models
'''

from abc import ABCMeta, abstractmethod
import numpy as np

from keras import optimizers

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
			self.label_dim = len(args['actions'])
			self.actions = args['actions']
			self.input_dim += self.label_dim

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

	@abstractmethod
	def make_model(self): # build model
		pass

	@abstractmethod
	def load_embedding(self, data, pred_only=False, reset=False):
		'''
		Populate self.embedding
		Args
			data: the data whose latent representations are used to populate the embedding
			pred_only: only populate what's relevant for prediction task
			reset: empty self.embedding and re-populat with data
		'''
		pass


	@abstractmethod
	def format_data(self, x, for_validation=False):
		'''
		Format the data to match the input and output of the model.
		'''
		pass

	@abstractmethod
	def predict(self, x, return_std=False):
		'''
		Motion prediction
		Args
			x: input data
			return_std: return std
		Return
			std, pred_y
		'''
		pass
