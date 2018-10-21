import os
import numpy as np
from sklearn.metrics import log_loss as sckit_log_loss

import converter

# For ouputs

OUTPUT_DIR = '../out/'

def create_dir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)

def output_dir(directory):
	dir = OUTPUT_DIR+directory
	create_dir(dir)
	return dir

# normalizations

def wrap_angle(rad, center=0):
	return ( rad - center + np.pi) % (2 * np.pi ) - np.pi

def norm_pi(rad, stats):
	center = stats['data_mean'][stats['dim_to_use']]
	return wrap_angle(rad, center)/np.pi

def unorm_pi(rad, stats):
	center = stats['data_mean'][stats['dim_to_use']]
	return wrap_angle(rad*np.pi, -center)

def norm_std(x, stats):
	mean, std = stats['data_mean'][stats['dim_to_use']], stats['data_std']
	return (x-mean)/std

def unorm_std(x, stats):
	mean, std = stats['data_mean'][stats['dim_to_use']], stats['data_std']
	return x*std+mean

def norm_max(x, stats):
	xmax, xmin = stats['data_min'], stats['data_max']
	return 2*(x-xmin)/(xmax-xmin)-1

def unorm_max(x, stats):
	xmax, xmin = stats['data_min'], stats['data_max']
	return (x+1)/2*(xmax-xmin)+xmin

def normalize(x, stats, norm_method):
	if norm_method != 'none':
		norm_x = np.copy(x)
		norm_x[:,:,:stats['data_dim']] =  (globals()[norm_method])(x[:,:,:stats['data_dim']], stats)
		return norm_x
	return x

def unormalize(x, stats, norm_method):
	if norm_method != 'none':
		unorm_x = np.copy(x)
		unorm_x[:,:,:stats['data_dim']] = (globals()['u'+norm_method])(x[:,:,:stats['data_dim']], stats)
		return unorm_x
	return x

def recover(x, stats):
	rec_x = np.zeros((x.shape[0], x.shape[1], stats['orig_data_dim']))
	rec_x[:,:,:] = stats['data_mean']
	rec_x[:,:,stats['dim_to_use']] = x[:,:,:stats['data_dim']]
	return rec_x

# error

SHORT_TERM_IDX = [1,3,7,9]
SHORT_N_LONG_TERM_IDX = [1,3,7,9,13,17,19,24]

def l2_error(x1, x2, averaged=True):
	'''
	This will return one number if dim = 2
	or will return a list if dim = 3
	'''
	s = np.sqrt(np.sum(np.square(x1[:,:,6:] - x2[:,:,6:]), -1))
	if averaged:
		return np.mean(s, 0)
	else:
		return s

def __convert_expmap2euler(x, stats):
	x = recover(x, stats)
	x_euler = np.zeros(x.shape)
	for i in range(x.shape[0]):
		x_euler[i] = converter.sequence_expmap2euler(x[i])
	x_euler[:,:,:6] = 0
	return x_euler

def prediction_error(y_pred, y_true, stats, averaged=True):
	if stats['parameterization'] == 'expmap':
		# similar to
		# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L203
		y_pred = __convert_expmap2euler(y_pred[:,:,:stats['data_dim']], stats)
		y_true = __convert_expmap2euler(y_true[:,:,:stats['data_dim']], stats)
		idx_to_use = np.where(np.std(np.reshape(y_true, (-1, y_true.shape[-1])), axis=0) > 1e-4)[0]
		return l2_error(y_pred[:,:,idx_to_use], y_true[:,:,idx_to_use], averaged)
	else:
		# use the same l2 for euclidean
		return l2_error(y_pred[:,:,:stats['data_dim']], y_true[:,:,:stats['data_dim']], averaged)

def softmax(x):
        '''
        bowored from https://gist.github.com/stober/1946926
        >>> res = softmax(np.array([0, 200, 10]))
        >>> np.sum(res)
        1.0
        >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
        True
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
        >>> np.sum(res, axis=1)
        array([ 1.,  1.,  1.])
        >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
        >>> np.sum(res, axis=1)
        array([ 1.,  1.])
        '''
	new_x = np.copy(x)
	for i in range(new_x.shape[0]):
	        max_x = np.max(new_x[i], axis=-1).reshape((-1, 1))
        	exp_x = np.exp(new_x[i] - max_x)
        	new_x[i] = exp_x / np.sum(exp_x, axis=-1).reshape((-1, 1))
        return new_x


def classification_error(y_pred, y_true, stats):
	if y_pred.shape[-1] > stats['data_dim']:
		y_pred = y_pred[:,:,stats['data_dim']:]
		y_true = y_true[:,:,stats['data_dim']:]
	return [sckit_log_loss(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]

# pretty print scores
def print_score(scores, title, keys, print_title=True, precision='.2'):
	# borrowed from
	# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/baselines.py#L190
	if title is not None:
		print '=== %s ==='%(title)

	def format_row(name, values, p=precision):
		s = '{0: <16}'.format(name)
		for i in SHORT_N_LONG_TERM_IDX:
			if i < len(values):
				s = s + (' | %'+p+'f')%(values[i])
			else:
				s = s + (' | n/a')
		return s

	if print_title:
		print format_row('milliseconds', [40*(i+1) for i in range(SHORT_N_LONG_TERM_IDX[-1]+1)], p='.0')
	for key in keys:
		print format_row(key, scores[key])
	# get average
	avg_score = np.mean(scores.values(), axis=0)
	print format_row('AVERAGE', avg_score)

def print_classification_score(score, actions):
	N = 8
	for action, i in actions.iteritems():
		s,e = i*N,(i+1)*N
		print action, np.mean(score[s:e])


if __name__ == '__main__':
	# unit testing
	a = (np.random.rand(2,3,4)*2-1)*np.pi
	stats = {'data_mean':np.zeros(4), 'dim_to_use':range(4), 'data_dim':4}
	b = normalize(a, stats, 'norm_pi')
	c = unormalize(b, stats, 'norm_pi')
	print a-c
	print euler_error(a,c, stats)
