import numpy as np

import keras.layers as K_layer
from keras.models import Model
from keras import backend as K

import abs_model

# https://bitbucket.org/parthaEth/humanposeprediction/src/1aa34f7d190b606d8c4c92dbca06e306cc53e08d/python_models/SupervisedGAN/SAM.py
# https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py
class ACGAN_RNN(abs_model.AbstractModel):

	def __init__(self, args):
		assert args['supervised']

		self.timesteps = args['timesteps']
		return super(ACGAN_RNN, self).__init__(args)

	def make_model(self):
		self.input_dim = self.input_dim - self.name_dim
		self.output_dim = self.name_dim

		noise = K_layer.Input(shape=(self.timesteps, self.input_dim))

		g_label = K_layer.Input(shape=(1,))
		g_label_ = K_layer.RepeatVector((self.timesteps*self.input_dim))(g_label)
		g_label_ = K_layer.Reshape((self.timesteps, self.input_dim))(g_label)

		gl = K_layer.Add()([noise, g_label_])
		gl = K_layer.TimeDistributed(Dense(self.latent_dim))(gl)
		gl = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(gl)
		gl = abs_model.RNN_UNIT(self.input_dim, return_sequences=True, activation=self.activation)(gl)

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		dl = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(inputs)
		dl = Dropout(0.98)(dl)
		d_label = K_layer.TimeDistributed(Dense(self.name_dim, activation='sigmoid'))(dl)
		validity = K_layer.TimeDistributed(Dense(1, activation="sigmoid"))(dl)

		self.encoder = Model([noise, g_label], gl, name='generator')
		self.decoder = Model(inputs, [validity, d_label], name='discriminator')

		loss = ['binary_crossentropy', 'sparse_categorical_crossentropy']
		self.decoder.compile(optimizer=self.opt, loss=loss)

		self.decoder.trainable = False
		self.model = Model(g_inputs, [validity, d_label], name='gan')
		self.model.compile(optimizer=self.opt, loss=loss)

	def load_embedding(self, data, **kwargs):
		# no embedding
		pass

	def format_data(self, x, **kwargs):
		return x[:,:,:-self.name_dim], x[:,:,-self.name_dim:]

	def predict(self, x, **kwargs):
		# not supported
		pass

	def classify(self, x, **kwargs):
		return self.decoder.predict(x)