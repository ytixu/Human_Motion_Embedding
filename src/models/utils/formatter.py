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
			#if model.add_noise:
			#	y[:,i,j] = y[:,i,j] + np.random.normal(0,model.noise_std,y[:,i,j].shape)
	y = np.reshape(y, (-1, model.timesteps*len(model.hierarchies), y.shape[-1]))
	return x, y

def expand_time_vl(model, x):
	'''
	Reformat the data so that we can encode sequences of different lengths.
	For VL_RNN
	'''
	n,t,d = x.shape
	new_x = np.zeros((n*len(model.hierarchies), t, d))

	for i, h in enumerate(model.hierarchies):
		# the first i frames
		new_x[i*n:(i+1)*n,:h+1] = x[:,:h+1]
		# the rest of the frames
		if model.repeat_last and h+1 != model.timesteps:
			for j in range(i*n,(i+1)*n):
				new_x[j,h+1:] = new_x[j,h]
		# TODO: add noise
	return new_x, new_x

def __pertube(model, data):
	#if model.add_noise:
	#	data[:] = np.random.normal(0,model.noise_std,data.shape)
	#else:
	data[:] = 0
	return data

def without_name(model, x):
	'''
	Return a copy of x without action name
	'''
	new_x = np.copy(x)
	new_x[:,:,-model.name_dim:] = 0
	return new_x

def without_motion(model, x):
	'''
	Return a copy of x without pose information
	'''
	new_x = np.copy(x)
	new_x[:,:,:-model.name_dim] = 0
	return new_x

def randomize_name(model, x):
	'''
	Randomly remove action name or pose sequence.
	Assume model.supervised == True.
	'''
	n = x.shape[0]
	rand_idx = np.random.choice(n, n*2/3, replace=False)
	n = len(rand_idx)/2
	# remove name
	__pertube(model, x[rand_idx[n:],:,-model.name_dim:])
	# remove pose
	__pertube(model, x[rand_idx[:n],:,:-model.name_dim])
	return x

EXPAND_NAMES_MODALITIES = ['motion', 'name', 'both']

def expand_names(model, x):
	new_x = {'both': np.copy(x),
			 'name': without_motion(model, x),
			 'motion': without_name(model, x)
	}
	return new_x, new_x

def format_h_rnn(model, x, **kwargs):
	if 'for_prediction' in kwargs and kwargs['for_prediction']:
		# remove the ground truth from input
		x_condition = np.copy(x)
		x_condition[:,model.timesteps_in:] = 0
		# ground truth is the motion after model.timesteps_in
		y = np.copy(x[:,model.timesteps_in:])

		return x_condition, y

	if 'for_classification' in kwargs and kwargs['for_classification']:
		# remove the name from input
		x_condition = without_name(model, x)
		# ground truth the motion name
		y = np.copy(x[:,:,-model.name_dim:])

		return x_condition, y

	random_name = False if 'expand_all_names' in kwargs and kwargs['expand_all_names'] else True

	if model.supervised and random_name:
		x = randomize_name(model, x)

	x,y = expand_time(model, x)

	if model.supervised and not random_name:
		y = np.concatenate([without_name(model, y), without_motion(model, y), y], axis=1)

	return x, y
