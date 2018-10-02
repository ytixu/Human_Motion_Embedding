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

def __label_dim(args):
	l = 0
	if args['supervised']:
		l = len(args['action_list'])
	return l

def __data_generator(data_dir, stats, args):
	'''
	Data generator for testing.
	Return only the used_dims + one hot label (if applicable)
	'''
	t = args['timesteps']
	for f in glob.glob(data_dir+'/test/*.npy'):
		data = np.load(f)[:,stats['dim_to_use']]
		n,d = data.shape
		n = n-t+1
		l = __label_dim(args)

		x = np.zeros((n, t, d+l))
		for i in range(n):
			# (txd)
			x[i,:,:d] = data[i:i+t,:]

		if args['supervised']:
			name = __get_action_from_file(f)
			x[:,:,-l+args['action_list'][name]] = 1

		yield x

def __data_generator_random(data_dir, stats, args):
	'''
	Data generator for training.
	Return only the used_dims + one hot label (if applicable)
	Similar to
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L435
	'''
	t,b = args['timesteps'], args['generator_size']
	data = {}
	sample_n = 0

	for i, f in enumerate(glob.glob(data_dir+'/train/*.npy')):
		data[i] = (np.load(f)[:,stats['dim_to_use']], __get_action_from_file(f))
		sample_n += 1

	conseq_n = 5
	l = __label_dim(args)
	x = np.zeros((b,t,stats['data_dim']+l))

	for i in range(args['iterations']):
		sample_idx = np.random.choice(sample_n, b/conseq_n)

		for j,sample_i in enumerate(sample_idx):
			sub_data, name = data[sample_i]
			n,d = sub_data.shape
			n = n - conseq_n - t + 1

			rand_idx = np.random.choice(n)
			for k in range(conseq_n):
				x[j*conseq_n+k,:,:d] = sub_data[rand_idx+k:rand_idx+k+t]

			if args['supervised']:
				name = __get_action_from_file(f)
				x[j*conseq_n:(j+1)*conseq_n,:,-l+args['action_list'][name]] = 1

		yield x

def __load_validation_data(data_dir, stats, args):
	'''
	Load validation data into one matrix.
	'''
	files = glob.glob(data_dir+'/test/*_cond.npy')
	l = __label_dim(args)
	d = stats['data_dim']
	ti,to = args['timesteps_in'],args['timesteps_out']
	x = np.zeros((len(files)*2,args['timesteps'],d+l))

	for i, f in enumerate(files):
		s,e = i*2,(i+1)*2
		x[s:e,:ti,:d] = np.load(f)[:,-ti:]
		x[s:e,-to:,:d] = np.load(f.replace('cond.npy','gt.npy')[:,:to]

	return x

# Get load and save path

def __format_file_name(a):
	return '-'.join('-'.join(
		a.strip('optimizers.').strip(')').split('(')).split(','))

def __get_model_path_name(args, file_type):
	ext, sub_dir = 'hdf5', '/models'
	if file_type == 'log':
		ext, sub_dir = 'csv', '/log'
	output_name = args['method_name'].lower()+sub_dir
	utils.output_dir(output_name)

	unit = 'gru'
	if args['lstm']:
		unit = 'lstm'

	opt_name = __format_file_name(args['optimizers'])
	return'%s/%s_t%d_l%d_u%d_loss-%s_opt-%s_%d.%s'%(
		output_name, unit, args['timesteps'], args['latent_dim'], args['unit_timesteps'],
		args['loss_func'], opt_name, time.time(), ext)

def get_parse(mode):
	ap = argparse.ArgumentParser()
	method_list = ['test', 'seq2seq']
	ap.add_argument('-m', '--method_name', required=True, help='Method name', choices=method_list)

	ap.add_argument('-id', '--input_data', required=False, help='Input data directory', default='../data/h3.6m/euler')
	# ap.add_argument('-od', '--output_data', required=False, help='Output data directory')
	ap.add_argument('-gs', '--generator_size', required=False, help='Size of the batch in the random data generator.', default=10000, type=int)
	whmtd_list = ['norm_pi', 'norm_std', 'norm_max', 'none']
	ap.add_argument('-w', '--normalization_method', required=False, help='Normalization method.', default='norm_pi', choices=whmtd_list)

	ap.add_argument('-lp', '--load_path', required=False, help='Model path')
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path')
	ap.add_argument('-log', '--log_path', required=False, help='Log file for loss history')

	ap.add_argument('-t', '--timesteps', required=False, help='Total timestep size', default=40, type=int)
	ap.add_argument('-ti', '--timesteps_in', required=False, help='Input timesteps', default=30, type=int)
	ap.add_argument('-to', '--timesteps_out', required=False, help='Output timesteps', default=10, type=int)
	ap.add_argument('-ut', '--unit_timesteps', required=False, help='Number of timesteps encoded at the first level', default=10, type=int)
	ap.add_argument('-iter', '--iterations', required=False, help='Number of iterations for training', default=int(1e5), type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default=64, type=int)
	ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default=800, type=int)
	ap.add_argument('-loss', '--loss_func', required=False, help='Loss function name', default='mean_absolute_error')
	ap.add_argument('-opt', '--optimizer', required=False, help='Optimizer and parameters', default='optimizers.Nadam(lr=0.001)')
	ap.add_argument('-lr', '--learning_rate', required=False, help='Learning rate', default=0.001, type=float)
	ap.add_argument('-lstm', '--lstm', action='store_true', help='Using LSTM instead of the default GRU')

	ap.add_argument('-sup', '--supervised', action='store_true', help='With action names')
	# ap.add_argument('-label', '--only_label', required=False, help='Only load data with this label', nargs = '*')

	args = vars(ap.parse_args())

	if 'save_path' not in args:
		args['save_path'] = __get_model_path_name(args, 'save')
	if 'log_path' not in args:
		args['log_path'] = __get_model_path_name(args, 'log')

	data_types = ['input_data']
	if 'output_data' in args:
		data_types = ['input_data', 'output_data']
	for t in data_types:
		stats = __get_stats(args[t])
		t = t+'_stats'
		args[t] = {
			'dim_to_use': stats['dim_to_use'],
			'data_mean': stats['data_mean'],
			'data_dim': len(stats['dim_to_use']),
			'orig_data_dim': len(stats['data_mean'])
		}
		for k in ['data_std', 'data_min', 'data_max']:
			args[t][k] = stats[k][stats['dim_to_use']]

	if args['supervised']:
		args['actions'] = stats['action_list']

	if mode == 'train':
		return args,
			__data_generator_random(args['input_data'], args['input_data_stats'], args),
			__load_validation_data(args['input_data'], args['input_data_stats'], args)

	assert('load_path' in args)
	return args, __data_generator(args['input_data'], args['input_data_stats'], args)


if __name__ == '__main__':
	for mode in ['train', 'test']:
		args, data_iter = get_parse(mode)
		print args
		for x in data_iter:
			print 'sample shape', x.shape
			break


