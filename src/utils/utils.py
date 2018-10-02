import os
import numpy as np

# For ouputs

OUTPUT_DIR = '../out/'

def create_dir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)

def output_dir(directory):
	create_dir(OUTPUT_DIR+directory)

# normalizations

def wrap_angle(rad, center=0):
	return ( rad - center + np.pi) % (2 * np.pi ) - np.pi

def norm_pi(rad, stats):
	center = stats['data_mean'][stat['dim_to_use']]
	return wrap_angle(rad, center)/np.pi

def unorm_pi(rad, stats):
	center = stats['data_mean'][stat['dim_to_use']]
	return wrap_angle(rad*np.pi, -center)

def norm_std(x, stats):
	mean, std = stats['data_mean'][stat['dim_to_use']], stats['data_std']
	return (x-mean)/std

def unorm_std(x, stats):
	mean, std = stats['data_mean'][stat['dim_to_use']], stats['data_std']
	return x*std+mean

def norm_max(x, stats):
	xmax, xmin = stats['data_min'], stats['data_max']
	return 2*(x-xmin)/(xmax-xmin)-1

def unorm_max(x, stats):
	xmax, xmin = stats['data_min'], stats['data_max']
	return (x+1)/2*(xmax-xmin)+xmin

def normalize(x, stats, norm_method):
	if norm_method != 'none':
		x[:,:,:stats['data_dim']] = (global()[norm_method])(x, stats)

def unormalize(x, stats, norm_method):
	if norm_method != 'none':
		x[:,:,:stats['data_dim']] = (global()['u'+norm_method])(x, stats)

def recover(x, stats):
	rec_x = np.zeros((x.shape[0], x.shape[1], stats['orig_data_dim']))
	rec_x[:,:] = self.data_mean
	rec_x[:,:,stats['dim_to_use']] = x
	return rec_x

# error

SHORT_TERM_IDX = [1,3,7,9]

def euler_error(x, y, stats):
	return np.mean(np.sqrt(np.sum(np.square(
		x[:,:,stats['dim_to_use']]], y[:,:,stats['dim_to_use']]]), -1)), 0)