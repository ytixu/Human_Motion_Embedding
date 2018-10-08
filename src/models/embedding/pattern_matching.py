import numpy as np
from scipy.spatial import distance

# pattern matching methods
ADD = 0
MEAN = 1
CLOSEST = 2
CLOSEST_PARTIAL = 3

# distance function
MSD = 0 # l2
MAD = 1 # l1
COS = 2 # cosine distance

def iter_methods():
	return ['add', 'mean', 'closest', 'closest_partial']

def iter_distance():
	return ['l2', 'l1', 'manhattan', 'cos']

def __distance__(e1, e2, dist_method=MAD):
	if dist_method == MSD:
		return np.linalg.norm(e1-e2)
	elif dist_method == MAD:
		return np.sum(np.abs(e1-e2))
	elif dist_method == COS:
		return distance.cosine(e1,e2)

def __get_dist(modality, z_ref, dist_method=MAD):
	return [__distance__(modality[i], z_ref, dist_method) for i in range(len(modality))]

def get_weights(modality, z_ref, dist_method=MAD):
	weights = __get_dist(modality, z_ref, dist_method)
	w_i = np.argsort(weights)
	return weights, w_i

def closest(modality, z_ref, weights=[], dist_method=MAD): #, return_weight=False):
	if not any(weights):
		weights = __get_dist(modality, z_ref, dist_method)
	idx = np.argmin(weights)
	# if return_weight:
	# 	return modality[idx], weights[idx]
	return modality[idx]

def __normalized_distance_mean(modality, weights, w_i):
	if weights[w_i[0]] < 1e-16 or weights[w_i[0]] == float('-inf'):
		return modality[w_i[0]]
	return np.sum([modality[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])

def closest_mean(modality, z_ref, n=5, weights=[], w_i=[], dist_method=MAD):
	if not any(weights):
		weights, w_i = get_weights(modality, z_ref, dist_method)
	if n > 0:
		w_i = w_i[:n]
	return __normalized_distance_mean(modality, weights, w_i)

def closest_random(modality, z_ref, n=-1, weights=[], w_i=[]):
	if n > 0:
		if not any(weights):
			modality = modality[np.random.choice(modality.shape[0], n, replace=False)]
		else:
			w_idx = sorted(np.random.choice(len(w_i), n, replace=False).tolist())
			w_i = [w_i[i] for i in w_idx]

	return closest_mean(modality, z_ref, weights=weights, w_i=w_i, dist_method=MAD)

def closest_partial(modality_partial, modality_complete, z_ref, weights=[], dist_method=MAD):
	if not any(weights):
		weights = __get_dist(modality_partial, z_ref, dist_method)
	idx = np.argmin(weights)
	return modality_complete[idx]

def __modality_diff(m_to, m_from):
	diff = m_to - m_from
	add_mean = np.mean(diff, axis=0)
	return add_mean, diff

def add(modality_partial, modality_complete, z_ref, return_std):
	add_mean, diff = __modality_diff(modality_complete, modality_partial)
	z_new = z_ref + add_mean
	if return_std:
		return np.std(diff, axis=0), z_new
	return z_new

def match(z_ref, model, **kwargs):
	'''
	Generic pattern matching method
	Args
		z_ref: reference pattern
		model: the RNN model
		kwargs:
			method=ADD: method to use
			modality_complete=model.embedding[model.timesteps-1]: the
					sub-embedding space where the predicted z will be in
			for ADD:
				modality_partial=model.embedding[model.timesteps_in-1]: the
					sub-embedding space where z_ref comes from
				return_std=False: whether to return the std
			for CLOSEST:
				dist_method=MAD: the distance method to use
				weights=[]: distance of each z in modality_complete to z_ref
			for MEAN:
				dist_method=MAD: the distance method to use
				weights=[]: distance of each z in modality_complete to z_ref
				w_i=[]: sorted indices in weights
				n=5: number of z in modality_complete to average on
			for CLOSEST_PARTIAL:
				dist_method=MAD: the distance method to use
				modality_partial=model.embedding[model.timesteps_in-1]: the
					sub-embedding space where z_ref comes from
				weights_to_pt=[]: distance of each z in modality_partial to z_ref
	'''
	method = kwargs['method'] if 'method' in kwargs else ADD
	dist = kwargs['dist_method'] if 'dist_method' in kwargs else MAD
	modality_complete = kwargs['modality_complete'] if 'modality_complete' in kwargs else model.embedding[model.timesteps-1]
	weights = kwargs['weights'] if 'weights' in kwargs else []

	if method == CLOSEST:
		return closest(modality_complete, z_ref, weights, dist)

	if method == MEAN:
		w_i = kwargs['w_i'] if 'w_i' in kwargs else []
		n = kwargs['n'] if 'n' in kwargs else 5
		return closest_mean(modality, z_ref, n, weights, w_i, dist)

	modality_partial = kwargs['modality_partial'] if 'modality_partial' in kwargs else model.embedding[model.timesteps_in-1]

	if method == CLOSEST_PARTIAL:
		weights_to_pt = kwargs['weights_to_pt'] if 'weights_to_pt' in kwargs else []
		return closest_partial(modality_partial, modality_complete, z_ref, weights_to_pt, dist)

	if method == ADD:
		return_std = kwargs['return_std'] if 'return_std' in kwargs else False
		return add(modality_partial, modality_complete, z_ref, return_std)

	raise ValueError('Unrecognized method %s'%(str(method)))

def raw_match(x_ref, model, **kwargs):
	'''
	Do pattern matching for raw data z_ref.
	WARNING except ADD, x_ref has 1 sample. TODO: change this later
	'''
	# encode
	c = model.timesteps_in-1
	z_ref = model.encode(x_ref, modality=c)
	# match
	z_matched = match(z_ref, model, kwargs)
	# decode
	x_matched = model.decode(z_matched)
	return x_matched

def batch_all_match(model, sample_zs):
	'''
	Assume embedding for model is already loaded. Used in test.py.

	Args
		model: the model
		sample_zs: the partial zs
	Yield
		matched_z
	'''

	c = model.timesteps-1
	modality_complete = model.embedding[c]
	modality_partial = model.embedding[model.timesteps_in-1]
	add_mean, diff = __modality_diff(modality_complete, modality_partial)

	kwargs = {
		'modality_complete': modality_complete
		'modality_partial' : modality_partial
	}

	for i in range(sample_zs.shape[0]):
		# select distance function
		for dist, dist_name in enumerate(iter_distance()):
			weights, w_i = get_weights(emb, sample_zs[i], dist_method=dist)
			kwargs['dist_method'] = dist
			kwargs['weights'] = weights
			kwargs['w_i'] = w_i

			# select method
			for method, mth_name in enumerate(iter_methods()[1:]): # skip add
				kwargs['method'] = method
				mth_name_ = mth_name

				if method == MEAN:
					for n in [5, 10, 50, 100]:
						kwargs['n'] = n
						mth_name_ = mth_name_+'-'+str(n)
						z_matched = match(sample_zs[i], model, kwargs)

				yield i, (dist_name, mth_name_), z_matched

		#add
		yield i, (None, 'add'), sample_zs[i] + add_mean