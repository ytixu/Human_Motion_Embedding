import numpy as np

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
		return OTHER_MODALITIES, kwargs['modalities']
	else:
		if 'pred_only' in kwargs and kwargs['pred_only']:
			sets = [model.timesteps_in-1, model.timesteps-1]
		return TIME_MODALITIES, model.hierarchies

