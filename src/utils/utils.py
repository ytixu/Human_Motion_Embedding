import os
import numpy as np

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
		norm_x[:,:,:stats['data_dim']] =  (globals()[norm_method])(x, stats)
		return norm_x
	return x

def unormalize(x, stats, norm_method):
	if norm_method != 'none':
		unorm_x = np.copy(x)
		unorm_x[:,:,:stats['data_dim']] = (globals()['u'+norm_method])(x, stats)
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

def euler_error(x, y, stats):
	return np.mean(np.sqrt(np.sum(np.square(
		x[:,:,:stats['data_dim']] - y[:,:,:stats['data_dim']]), -1)), 0)

def list_short_term(model, error):
	idx = [model.timesteps_in + i for i in SHORT_TERM_IDX]
	return error[idx]

if __name__ == '__main__':
	# unit testing
	a = (np.random.rand(2,3,4)*2-1)*np.pi
	stats = {'data_mean':np.zeros(4), 'dim_to_use':range(4), 'data_dim':4}
	b = normalize(a, stats, 'norm_pi')
	c = unormalize(b, stats, 'norm_pi')
	print a-c
	print euler_error(a,c, stats)
