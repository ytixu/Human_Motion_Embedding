import numpy as np
import time
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import cross_validation
from keras import optimizers
import keras.backend as K_backend

class Forward_NN:

	def __init__(self, args):
		self.batch_size = args['batch_size']
		self.epochs = 5
		self.embedding_dim = args['latent_dim']

		self.lr = args['learning_rate']
		self.decay = args['decay']
		self.loss_func = args['loss_func']
		self.opt = eval(args['optimizer'])

		self.interim_dim = self.embedding_dim

		self.model = None
		self.make_model()


	def make_model(self):
		inputs = Input(shape=(self.embedding_dim,))
		#d1 = Dense(self.interim_dim, activation='relu')(inputs)
		#d2 = Dense(self.interim_dim, activation='relu')(d1)
		#d3 = Dense(self.interim_dim, activation='relu')(d2)
		outputs = Dense(self.embedding_dim, activation='tanh')(inputs)

		# def mse(yTrue, yPred):
		# 	return K_backend.mean(K_backend.sqrt(K_backend.sum(K_backend.square(yTrue - yPred), axis=0)))

		self.model = Model(inputs, outputs)
		self.model.compile(optimizer=self.opt(lr=self.lr), loss=self.loss_func)
		self.model.summary()

	def load(self, load_path):
		self.model.load_weights(load_path)

	def decay_learning_rate(self):
		if self.decay < 1:
			self.lr = self.lr * self.decay
			self.model.compile(optimizer=self.opt(lr=self.lr), loss=self.loss_func)

	def predict(self, x):
		return self.model.predict(x)
