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
	if ('reset' in kwargs and kwargs['reset']):
		model.reset_embedding()

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
