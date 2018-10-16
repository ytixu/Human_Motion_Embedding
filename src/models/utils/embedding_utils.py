import numpy as np
import formatter

TIME_MODALITIES = 0
OTHER_MODALITIES = 1

def parse_load_embedding(model, data, **kwargs):
	'''
	Common routines for VL_RNN, H_RNN or HH_RNN
	Args
		model: RNN model (either VL_RNN, H_RNN or HH_RNN)
		data: data to populate the embedding
		kwargs:
			reset=False: whether or not to reset the model's embedding
						space and re-populate it
			pred_only=False: populate only the relevant time-modalities
						for motion prediction task
			modalities=model.hierarchies: the relevant modalities that we want
						to populate
	Return
		TIME_MODALITIES or OTHER_MODALITIES
		set: relevant modalities
	'''

	# reset embedding
	if ('reset' in kwargs and kwargs['reset']) or model.embedding is None:
		model.embedding = {}

	# select relevant modalities
	if 'modalities' in kwargs:
		# assume data is already key-ed by the modalities
		return OTHER_MODALITIES, kwargs['modalities'], data

	# load for classification task
	elif 'class_only' in kwargs and kwargs['class_only']:
		sets = formatter.EXPAND_NAMES_MODALITIES
		data,_ = formatter.expand_names(model, data)
		return OTHER_MODALITIES, sets, data

	# load for motion prediction task
	elif 'pred_only' in kwargs and kwargs['pred_only']:
		return TIME_MODALITIES, [model.timesteps_in-1, model.timesteps-1], data

	# load all time-modalities
	return TIME_MODALITIES, model.hierarchies, data


def interpolate(z_a, z_b, l=8):
        dist = (z_b - z_a)/l
        return np.array([z_a+i*dist for i in range(l+1)])

def softmax(x):
	'''
	bowored from https://gist.github.com/stober/1946926
	>>> res = softmax(np.array([0, 200, 10]))
	>>> np.sum(res)
	1.0
	>>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
	True
	>>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
	>>> np.sum(res, axis=1)
	array([ 1.,  1.,  1.])
	>>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
	>>> np.sum(res, axis=1)
	array([ 1.,  1.])
	'''
	b,t,d = x.shape
	x = x.reshape((-1, d))
	max_x = np.max(x, axis=1).reshape((-1, 1))
	exp_x = np.exp(x - max_x)
	x = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
	return x.reshape((b,t,d))
