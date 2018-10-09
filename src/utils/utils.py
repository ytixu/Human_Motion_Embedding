import os
import numpy as np

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
	rec_x[:,:] = self.data_mean
	rec_x[:,:,stats['dim_to_use']] = x
	return rec_x

# error

SHORT_TERM_IDX = [1,3,7,9]
LONG_TERM_IDX = [11,13,15,17,19,21,23,24]

def l2_error(x1, x2):
	'''
	This will return one number if dim = 2
	or will return a list if dim = 3
	'''
	return np.mean(np.sqrt(np.sum(np.square(x1 - x2), -1)), 0)

def __convert_expmap2euler(x, stats):
	x = recover(x, stats)
	x = converter.sequence_expmap2euler(y_true)
	x[:,:,:6] = 0
	return x

def prediction_error(y_pred, y_true, stats):
	if stats['parameterization'] == 'expmap':
		# similar to
		# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L203
		y_pred = __convert_expmap2euler(y_pred[:,:,:stats['data_dim']])
		y_true = __convert_expmap2euler(y_true[:,:,:stats['data_dim']])
		idx_to_use = np.where(np.std(np.reshape(y_true, (-1, y_true.shape[-1])), axis=0) > 1e-4)[0]
		print idx_to_use #TODO: remove
		return l2_error(y_pred[:,:,idx_to_use], y_true[:,:,idx_to_use])
	else:
		# use the same l2 for euclidean
		return l2_error(y_pred[:,:,:stats['data_dim']], y_true[:,:,:stats['data_dim']])

def list_short_term(model, error):
	idx = [model.timesteps_in + i for i in SHORT_TERM_IDX]
	return error[idx]

# pretty print scores
def print_short_term_score(scores, name, keys):
	# borrowed from
	# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/baselines.py#L190
	print '=== %s ==='%(name)
	print '{0: <16} | {1:4d} | {2:4d} | {3:4d} | {4:4d}'.format('milliseconds', 80, 160, 380, 400)
	for key in keys:
		print '{0: <16} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f}'.format( action,
			scores[key][1], scores[key][3], scores[key][7], scores[key][9] )
	# get average
	avg_score = np.mean(scores.values(), axis=0)
	print '{0: <16} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f}'.format( 'AVERAGE',
		avg_score[1], avg_score[3], avg_score[7], avg_score[9] )


if __name__ == '__main__':
	# unit testing
	a = (np.random.rand(2,3,4)*2-1)*np.pi
	stats = {'data_mean':np.zeros(4), 'dim_to_use':range(4), 'data_dim':4}
	b = normalize(a, stats, 'norm_pi')
	c = unormalize(b, stats, 'norm_pi')
	print a-c
	print euler_error(a,c, stats)
