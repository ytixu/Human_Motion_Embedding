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
		g_label_ = K_layer.Reshape((self.timesteps, self.input_dim))(g_label_)

		gl = K_layer.Add()([noise, g_label_])
		gl = K_layer.TimeDistributed(K_layer.Dense(self.latent_dim))(gl)
		gl = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)(gl)
		gl = abs_model.RNN_UNIT(self.input_dim, return_sequences=True, activation=self.activation)(gl)

		inputs = K_layer.Input(shape=(self.timesteps, self.input_dim))
		dl_1 = abs_model.RNN_UNIT(self.latent_dim, return_sequences=True)
		dl_2 = K_layer.Dropout(0.98)
		dl_3 = K_layer.TimeDistributed(K_layer.Dense(1, activation="sigmoid"))
		dl_4 = K_layer.TimeDistributed(K_layer.Dense(self.name_dim))
		dl_5 = K_layer.Lambda(lambda x: K.tf.nn.softmax(x))

		dl_ = dl_2(dl_1(inputs))
		validity_ = dl_3(dl_)
		d_label_ = dl_5(dl_4(dl_))

		self.encoder = Model([noise, g_label], gl, name='generator')
		self.decoder = Model(inputs, [validity_, d_label_], name='discriminator')

		self.loss = ['binary_crossentropy', 'categorical_crossentropy']
		self.decoder.compile(optimizer=self.opt, loss=loss)

		self.decoder.trainable = False
		validity, d_label = self.decoder(self.encoder([noise, g_label]))
		self.model = Model([noise, g_label], [validity, d_label], name='gan')

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
