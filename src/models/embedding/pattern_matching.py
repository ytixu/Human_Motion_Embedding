import numpy as np
from scipy.spatial import distance

# pattern matching methods
ADD = 0
MEAN = 1
CLOSEST = 2

# distance function
MSD = 0 # l2
MAD = 1 # l1
COS = 2 # cosine distance


def __dist_name__(idx):
	return ['l2', 'l1', 'manhattan', 'cos'][idx]

def __distance__(e1, e2, mode=1):
	if mode == 0:
		return np.linalg.norm(e1-e2)
	elif mode == 1:
		return np.sum(np.abs(e1-e2))
	elif mode == 2:
		return distance.cosine(e1,e2)

def __get_dist(modality, z_ref, mode=1):
	return [__distance__(modality[i], z_ref, mode) for i in range(len(modality))]

def get_weights(modality, z_ref, mode=1):
	weights = __get_dist(modality, z_ref, mode)
	w_i = np.argsort(weights)
	return weights, w_i

def closest(modality, z_ref, weights=[], return_weight=False):
	if not any(weights):
		weights = __get_dist(modality, z_ref)
	idx = np.argmin(weights)
	if return_weight:
		return modality[idx], weights[idx]
	return modality[idx]

def __normalized_distance_mean(modality, weights, w_i):
	if weights[w_i[0]] < 1e-16 or weights[w_i[0]] == float('-inf'):
		return modality[w_i[0]]
	return np.sum([modality[d]/weights[d] for d in w_i], axis=0)/np.sum([1.0/weights[d] for d in w_i])

def closest_mean(modality, z_ref, n=5, weights=[], w_i=[]):
	if not any(weights):
		weights, w_i = get_weights(modality, z_ref)
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

	return closest_mean(modality, z_ref, weights=weights, w_i=w_i)

def closest_partial_index(modality_partial, z_ref, weights={}):
	if not any(weights):
		weights = __get_dist(modality_partial, z_ref)
	return np.argmin(weights)

def add(modality_partial, modality_complete, z_ref, return_std):
	# assume
	diff = modality_partial - modality_complete
	add_mean = np.mean(diff, axis=0)
	z_new = z_ref + add_mean
	if return_std:
		return np.std(diff, axis=0), z_new
	return z_new


def compare(model, data):
	pass