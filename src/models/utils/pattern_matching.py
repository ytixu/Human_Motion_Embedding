import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

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
	return ['add'] #, 'mean', 'closest', 'closest_partial']

def iter_distance():
	return ['l2', 'l1', 'cos']

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

def closest(modality, z_ref, weights=[], dist_method=MAD):
	if not any(weights):
		weights = __get_dist(modality, z_ref, dist_method)
	idx = np.argmin(weights)
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

def __modality_diff(m_from, m_to):
	diff = m_to - m_from
	add_mean = np.mean(diff, axis=0)
	return add_mean, diff

def add(modality_partial, modality_complete, z_ref, return_std):
	add_mean, diff = __modality_diff(modality_partial, modality_complete)
	z_new = z_ref + add_mean
	if return_std:
		return np.std(diff, axis=0), z_new
	return z_new

def noisy_add(modality_partial, modality_complete, z_ref, return_std):
	add_mean, diff = __modality_diff(modality_partial, modality_complete)
	std = np.std(diff, axis=0)
	z_new = z_ref + np.random.normal(add_mean, std, z_ref.shape)
	if return_std:
		return std, z_new
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
		return closest_mean(modality_complete, z_ref, n, weights, w_i, dist)

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
	Args
		x_ref:
		model:
		kwargs:
			partial_encode_idx=model.timesteps_in-1: index for retrieving the partial sequence encoding
				(only useful for H_RNN and HH_RNN)
			return_seq_fn=identity: extract the relevant pattern as output
			... other params in match()
	Return
		same as match() but the matched pattern is decoded
	'''
	fn = kwargs['return_seq_fn'] if 'return_seq_fn' in kwargs else lambda x:x
	# encode
	c = kwargs['partial_encode_idx'] if 'partial_encode_idx' in kwargs else model.timesteps_in-1
	z_ref = model.encode(x_ref, modality=c)
	# match
	z_matched = match(z_ref, model, **kwargs)
	# decode
	if 'return_std' in kwargs and kwargs['return_std']:
		std, z_matched = z_matched
		return std, fn(model.decode(z_matched))
	return fn(model.decode(z_matched))

def batch_all_match(model, sample_zs, modalities):
	'''
	Assume embedding for model is already loaded. Used in test.py.

	Args
		model: the model
		sample_zs: the partial zs
		modalities=(partial_idx, complete_idx)
	Yield
		matched_z
	'''

	modality_partial = model.embedding[modalities[0]]
	modality_complete = model.embedding[modalities[1]]
	add_mean, diff = __modality_diff(modality_partial, modality_complete)
	std = np.std(diff, 0)

	kwargs = {'modality_partial' : modality_partial,
			  'modality_complete': modality_complete
	}

	for i in tqdm(range(sample_zs.shape[0])):
		# select distance function
		for dist, dist_name in enumerate(iter_distance()):
			#TODO: remove this
			#if dist_name in ['l2','cos']: continue

			weights, w_i = get_weights(modality_complete, sample_zs[i], dist_method=dist)
			kwargs['dist_method'] = dist
			kwargs['weights'] = weights
			kwargs['w_i'] = w_i

			# select method
			for method, mth_name in enumerate(iter_methods()[1:]): # skip add
				kwargs['method'] = method+1

				if method+1 == MEAN:
					for n in [50]: # [5,50,100,500,1000]:
						kwargs['n'] = n
						mth_name_ = mth_name+'-'+str(n)
						z_matched = match(sample_zs[i], model, **kwargs)
						yield i, '%s(%s)'%(mth_name_,dist_name), z_matched
				else:
					z_matched = match(sample_zs[i], model, **kwargs)
					yield i, '%s(%s)'%(mth_name,dist_name), z_matched

		# add
		yield i, 'add', sample_zs[i] + add_mean
		for noise in range(1,6):
			yield i, 'noisy_add_%d'%noise, sample_zs[i] + np.random.normal(add_mean, std/noise, sample_zs[i].shape)
