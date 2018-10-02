import argparse
import glob
import json
import os.path
import time

import numpy as np

import utils

# Get data and information on the data

def __get_stats(data_dir):
	return json.load(open(data_dir+'/stats.json'))

def __get_action_from_file(f):
	return os.path.basename(f).split('_')[0]

def __data_generator(data_dir, stat, args):
	'''
	Data generator for testing.
	Return only the used_dims + one hot label (if applicable)
	'''
	t = args['timesteps']
	for f in glob.glob(data_dir+'/test/'):
		data = np.load(f)[:,stat['dim_to_use']]
		n,d = data.shape
		n = n-t+1
		l = 0
		if args['supervised']:
			l = len(args['action_list'])

		x = np.zeros((n, t, d+l))
		for i in range(n):
			# (txd)
			x[i,:,:d] = data[i:i+t,:]

		if args['supervised']:
			name = __get_action_from_file(f)
			x[:,:,-l+args['action_list'][name]] = 1

		yield x

def __data_generator_random(data_dir, stat, args):
	'''
	Data generator for training.
	Return only the used_dims + one hot label (if applicable)
	Similar to
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L435
	'''
	t,b = args['timesteps'], args['generator_size']
	data = {}
	sample_n = 0
	files = glob.glob(data_dir+'/train/')

	for i, f in enumerate(files):
		data[i] = (np.load(f)[:,stat['dim_to_use']], __get_action_from_file(f))
		sample_n += 1

	files = [__get_action_from_file(files)]

	conseq_n = 5
	l = 0
	if args['supervised']:
		l = len(args['action_list'])
	x = np.zeros((b,t,args['data_dim']+l))

	for i in range(args['iterations']):
		sample_idx = np.random.choice(sample_n, b/conseq_n)

		for j,sample_i in enumerate(sample_idx):
			n,d = data[sample_i].shape
			n = n - conseq_n - t + 1

			rand_idx = np.random.choice(n)
			for k in range(conseq_n):
				x[j*conseq_n+k,:,:d] = data[sample_i][rand_idx+k:rand_idx+k+t]

			if args['supervised']:
				name = __get_action_from_file(f)
				x[j*conseq_n:(j+1)*conseq_n,:,-l+args['action_list'][files[sample_i]]] = 1

		yield x



# Get load and save path

def __format_file_name(a):
	return '-'.join('-'.join(
		a.strip('optimizers.').strip(')').split('(')).split(','))

def __get_model_path_name(model_name, args, file_type):
	ext, sub_dir = 'hdf5', '/models'
	if file_type == 'log':
		ext, sub_dir = 'csv', '/log'
	output_name = model_name.lower()+sub_dir
	utils.output_dir(output_name)

	opt_name = __format_file_name(args['optimizers'])
	return'%s/t%d_l%d_u%d_loss-%s_opt-%s_%d.%s'%(
		output_name, args['timesteps'], args['latent_dim'], args['unit_timesteps']
		args['loss_func'], opt_name, time.time(), ext)

def get_parse(model_name, mode):
	ap = argparse.ArgumentParser()
	ap.add_argument('-id', '--input_data', required=False, help='Input data directory', default='../data/h3.6m/euler')
	# ap.add_argument('-od', '--output_data', required=False, help='Output data directory')
	ap.add_argument('-gs', '--generator_size', required=False, help='Size of the batch in the random data generator.', default=10000, type=int)
	whmtd_list = ['norm_pi', 'norm_std', 'norm_max', 'none']
	ap.add_argument('-w', '--whitening_method', required=False, help='Whitening method.', type='norm_pi', choices=whmtd_list)

	ap.add_argument('-lp', '--load_path', required=False, help='Model path')
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path')
	ap.add_argument('-log', '--log_path', required=False, help='Log file for loss history', default=get_log_name(model_name))

	ap.add_argument('-t', '--timesteps', required=False, help='Total timestep size', default=40, type=int)
	ap.add_argument('-ut', '--unit_timesteps', required=False, help='Number of timesteps encoded at the first level', default=10, type=int)
	ap.add_argument('-iter', '--iterations', required=False, help='Number of iterations for training', default=int(1e5), type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default=64, type=int)
	ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default=800, type=int)
	ap.add_argument('-loss', '--loss_func', required=False, help='Loss function name', default='mean_absolute_error')
	ap.add_argument('-opt', '--optimizer', required=False, help='Optimizer and parameters', default='optimizers.Nadam(lr=0.001)')
	ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default=0.001, type=float)

	ap.add_argument('-sup', '--supervised', action='store_true', help='With action names')
	# ap.add_argument('-label', '--only_label', required=False, help='Only load data with this label', nargs = '*')

	args = vars(ap.parse_args())

	if 'save_path' not in args:
		args['save_path'] = __get_model_path_name(model_name, args, 'save')
	if 'log_path' not in args:
		args['log_path'] = __get_model_path_name(model_name, args, 'log')

	data_types = ['input_data']
	if 'output_data' in args:
		data_types = ['input_data', 'output_data']
	for t in data_types:
		stats = __get_stats(args[t])
		args[data_type+'_stats'] = {}
		for k in ['data_mean', 'data_std', 'dim_to_use', 'data_dim']:
			args[t][k] = stats[k]

	if args['supervised']:
		args['actions'] = stats['action_list']

	if mode == 'train':
		return args, __data_generator_random(args['input_data'], args['input_data_stats'], args)

	return args, __data_generator(args['input_data'], args['input_data_stats'], args)


if __name__ == '__main__':
	args, data_iter = get_parse('TEST', 'train')
	print args

	for x in data_iter:
		print x.shape

