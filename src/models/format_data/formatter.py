import numpy as np

def expand_time(model, x):
	'''
	Get different time-modalities.
	For H_RNN, HH_RNN.
	'''
	y = np.repeat(x, len(model.hierarchies), axis=0)
	y = np.reshape(y, (-1, len(model.hierarchies), model.timesteps, y.shape[-1]))
	# change the irrelevant frames
	for i, h in enumerate(model.hierarchies):
		for j in range(h+1, model.timesteps):
			if model.repeat_last:
				y[:,i,j] = y[:,i,h]
			else:
				y[:,i,j] = 0
	y = np.reshape(y, (-1, model.timesteps*len(model.hierarchies), y.shape[-1]))
	return x, y

def randomize_time(model, x):
	'''
	Randomly remove frames. For VL_RNN.
	'''
	n,t,d = x.shape
	rand_ts = np.random.choice(t, n)
	new_x = np.copy(x)

	for i, idx in enumerate(rand_ts):
		if idx+1 != self.timesteps:
			if self.repeat_last:
				new_x[i,idx+1:] = x[i,idx]
			else:
				new_x[i,idx+1:] = 0

	return new_x, new_x

def randomize_label(model, x):
	'''
	Randomly remove action name or pose sequence.
	Assume model.supervised == True.
	'''
	n = x.shape[0]
	rand_idx = np.random.choice(n, n*2/3, replace=False)
	n = len(rand_idx)/2
	# remove name
	x[rand_idx[n:],:,-model.label_dim:] = 0
	# remove pose
	x[rand_idx[:n],:,:-model.label_dim] = 0
	return x

def expand_modalities(model, x, for_validation=False):
	if for_validation:
		# remove the ground truth that we are trying to predict
		x_condition = np.copy(x)
		x_condition[:,model.timesteps_in:] = 0

		return x_condition, x

	if model.supervised:
		x = randomize_label(model, x)
	return expand_time(model, x)