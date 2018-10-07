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
	Randomly expand time-modalities.
	For VL_RNN.
	'''
	n,t,d = x.shape
	rand_idx = np.random.choice(n, model.expand_t, replace=False)
	new_x = np.zeros((n*rand_idx, t, d))

	for i, idx in enumerate(rand_idx):
		# the first i frames
		new_x[i*n:(i+1)*n,:idx+1] = x[:,:idx+1]
		# the rest of the frames
		if self.repeat_last and idx+1 != self.timesteps:
			for j in range(i*n,(i+1)*n):
				new_x[j,idx+1:] = x[j,idx]
	return new_x, new_x

def randomize_label(model, x):
	'''
	Randomly remove half of the labels.
	Assume model.supervised == True.
	'''
	n = x.shape[0]
	rand_idx = np.random.choice(n, n/2, replace=False)
	x[rand_idx,:,-model.label_dim:] = 0
	return x
