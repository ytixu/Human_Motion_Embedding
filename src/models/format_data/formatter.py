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

def expand_time(model, x):
	'''
	Reformat the data so that we can encode sequences of different lengths.
	'''
	n,t,d = x.shape
	new_x = np.zeros((n*model.timesteps, t, d))

	for i in range(model.timesteps):
		# the first i frames
		new_x[i*n:(i+1)*n,:i+1] = x[:,:i+1]
		# the rest of the frames
		if model.repeat_last and i+1 != model.timesteps:
			for j in range(i*n,(i+1)*n):
				new_x[j,i+1:] = new_x[j,i]
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
