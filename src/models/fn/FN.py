import numpy as np
import time
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import cross_validation
from keras.optimizers import Nadam as OptAlgo
import keras.backend as K_backend

import parser

NAME = 'Forward_NN'

class Forward_NN:

	def __init__(self, args):
		self.batch_size = args['batch_size'] if 'batch_size' in args else 16
		self.epochs = args['epochs'] if 'epochs' in args else 5
		self.output_dim = args['output_dim']
		self.input_dim = args['input_dim']
		self.interim_dim = (args['output_dim'] + args['input_dim'])/2
		self.layers = args['layers'] if 'layers' in args else 1
		self.epsilon_std = 1.0
		self.periods = args['periods'] if 'periods' in args else 10
		self.cv_splits = args['cv_splits'] if 'cv_splits' in args else 0.2
		self.trained = args['mode'] == 'sample'

		self.save_path = args['save_path']

		self.model = None

	def make_model(self):
		inputs = Input(shape=(self.input_dim,))
		#d1 = Dense(self.interim_dim, activation='relu')(inputs)
		#d2 = Dense(self.interim_dim, activation='relu')(d1)
		#d3 = Dense(self.interim_dim, activation='relu')(d2)
		outputs = Dense(self.output_dim, activation='tanh')(inputs)

		def mse(yTrue, yPred):
			return K_backend.mean(K_backend.sqrt(K_backend.sum(K_backend.square(yTrue - yPred), axis=0)))

		self.model = Model(inputs, outputs)
		opt = OptAlgo(lr=0.001)
		self.model.compile(optimizer=opt, loss=mse)
		self.model.summary()

	def load(self, load_path):
		self.make_model()
		if self.trained:
			self.model.load_weights(load_path)
			print 'loaded'
			return True
		return False


	def run(self, data_iterator):
		if not self.load():
			loss = 10000
			lr = 0.001
			count = 0
			for x, y in data_iterator:
				x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=self.cv_splits)
				history = self.model.fit(x_train, y_train,
							shuffle=True,
							epochs=self.epochs,
							batch_size=self.batch_size,
							validation_data=(x_test, y_test))
							# callbacks=[self.history])

				new_loss = np.mean(history.history['loss'])
				if new_loss < loss:
					print 'Saved model - ', loss
					loss = new_loss
					self.model.save_weights(self.save_path, overwrite=True)
					count = 0
				elif count > 3:
					lr = lr/2
				 	opt = OptAlgo(lr=lr)
				 	self.model.compile(optimizer=opt, loss='mean_squared_error')
				 	print 'new learning rate', lr
					count = 0
				else:
					count += 3
					print count, 'failed'

if __name__ == '__main__':
	data_iterator, config = parser.get_parse(NAME)
	nn = Forward_NN(config)
	nn.run(data_iterator)
