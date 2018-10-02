import argparse
import time
import glob
import os.path
import numpy as np

import utils

# Get load and save path

def __format_file_name(a):
	return '-'.join('-'.join(
		a.strip('optimizers.').strip(')').split('(')).split(','))

def get_model_path_name(model_name, args, file_type):
	ext, sub_dir = 'hdf5', '/models'
	if file_type == 'log':
		ext, sub_dir = 'csv', '/log'
	output_name = model_name.lower()+sub_dir
	utils.output_dir(output_name)

	opt_name = __format_file_name(args['optimizers'])
	return'%s/t%d_l%d_u%d_loss-%s_opt-%s_%d.%s'%(
		output_name, args['timesteps'], args['latent_dim'], args['unit_timesteps']
		args['loss_func'], opt_name, time.time(), ext)

# Get list of labels with the index for the 1-hot encoding
def get_one_hot_labels(files):
	labels = set(map(lambda x: os.path.basename(x).split('_')[0], files))
	return {l:i for i,l in enumerate(labels)}, len(labels)


def get_parse(model_name, labels=False):
	ap = argparse.ArgumentParser()
	ap.add_argument('-id', '--input_data', required=True, help='Input data directory')
	ap.add_argument('-od', '--output_data', required=False, help='Output data directory')
	ap.add_argument('-tid', '--test_input_data', required=False, help='Test input data directory')
	ap.add_argument('-tod', '--test_output_data', required=False, help='Test output data directory')

	ap.add_argument('-lp', '--load_path', required=False, help='Model path')
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path')
	ap.add_argument('-log', '--log_path', required=False, help='Log file for loss history', default=get_log_name(model_name))

	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='1', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='64', type=int)
	ap.add_argument('-t', '--timesteps', required=False, help='Total timestep size', default='40', type=int)
	ap.add_argument('-ut', '--unit_timesteps', required=False, help='Number of timesteps encoded at the first level', default='10', type=int)
	ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='1', type=int)
	ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default='800', type=int)
	ap.add_argument('-loss', '--loss_func', required=False, help='Loss function name', default='mean_absolute_error')
	ap.add_argument('-opt', '--optimizer', required=False, help='Optimizer and parameters', default='optimizers.Nadam(lr=0.001)')
	ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default='0.001', type=float)

	ap.add_argument('-sup', '--supervised', action='store_true', help='With action names')
	ap.add_argument('-label', '--only_label', required=False, help='Only load data with this label', nargs = '*')

	args = vars(ap.parse_args())

	if 'save_path' not in args:
		args['save_path'] = get_model_path_name(model_name, args, 'save')
	if 'log_path' not in args:
		args['log_path'] = get_model_path_name(model_name, args, 'log')

	if args['supervised']:

