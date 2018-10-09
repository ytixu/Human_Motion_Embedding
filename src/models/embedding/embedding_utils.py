import numpy as np

def load_embedding(model, data, **kwargs):
	'''
	Populate the embedding space with data
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
	'''
	# reset embedding
	if ('reset' in kwargs and kwargs['reset']) or model.embedding is None:
		model.embedding = {}

	# select relevant modalities
	sets = model.hierarchies
	if 'modalities' in kwargs:
		sets = kwargs['modalities']
		# assume data is key-ed by the modalities TODO: remove this
	else:
		if 'pred_only' in kwargs and kwargs['pred_only']:
			sets = [model.timesteps_in-1, model.timesteps-1]

		# assume data is formatted according to model.format_data()
		# reformat it so that it is key-ed by the modalities TODO: remove this
		n,k,d = data.shape
		t = len(model.hierarchies)
		data = np.reshape(data, (t,n/t,k,d))
		data = {h:data[i] for i,h in enumerate(model.hierarchies)}

	# populate
	for i in sets:
		zs = model.encoder.predict(data[i])
		if i not in model.embedding:
			model.embedding[i] = zs
		else:
			model.embedding[i] = np.concatenate([model.embedding[i], zs])
