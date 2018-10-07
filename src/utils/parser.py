import argparse
import glob
import json
import os.path
import time

import numpy as np

import utils

# limit memory size
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

METHOD_LIST = ['test', 'Seq2Seq', 'VL_RNN', 'H_RNN', 'HH_RNN', 'H_Seq2Seq']

# Get data and information on the data

def __get_stats(data_dir):
	return json.load(open(data_dir+'/stats.json'))

def __get_action_from_file(f):
	return os.path.basename(f).split('_')[0]

def __label_dim(args):
	l = 0
	if args['supervised']:
		l = len(args['actions'])
	return l

def __data_generator(data_dir, stats, args):
	'''
	Data generator for testing.
	Return only the used_dims + one hot label (if applicable)
	'''
	t = args['timesteps']
	for f in glob.glob(data_dir+'*.npy'):
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
			x[:,:,-l+args['actions'][name]] = 1

		yield x[:100]

def __data_generator_random(data_dir, stats, args, b):
	'''
	Data generator for training.
	Return only the used_dims + one hot label (if applicable)
	Similar to
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L435
	'''
	t = args['timesteps']
	data = {}
	sample_n = 0

	for i, f in enumerate(glob.glob(data_dir+'*.npy')):
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
				x[j*conseq_n:(j+1)*conseq_n,:,-l+args['actions'][name]] = 1

		yield x

def __load_validation_data(data_dir, stats, args):
	'''
	Load validation data into one matrix.
	'''
	files = glob.glob(data_dir+'/valid/*-cond.npy')
	l = __label_dim(args)
	d = stats['data_dim']
	ti,to = args['timesteps_in'],args['timesteps_out']
	x = np.zeros((len(files)*4,args['timesteps'],d+l))

	for i, f in enumerate(files):
		s,e = i*4,(i+1)*4
		x[s:e,:ti,:d] = np.load(f)[:,-ti:,stats['dim_to_use']]
		x[s:e,-to:,:d] = np.load(f.replace('cond.npy','gt.npy'))[:,:to,stats['dim_to_use']]

	return x

# Get load and save path

def __format_file_name(a):
	return '-'.join('-'.join(
		a.strip(')').split('(')).split(','))

def __get_model_path_name(args, file_type):
	'''
	Format name for the log and model file. Name include:
		directory: 	model name
					parameterization (e.g.: euler)
					supervised
		filename: 	rnn unit (lstm or gru)
					timesteps
					latent_dim
					unit_timesteps
					loss_func
					optimizer
					normalization_method
					repeat_last
					timestamp
		extension:  .log or .hdf5
	'''
	ext = 'hdf5'
	if file_type == 'log':
		ext = 'csv'

	output_name = args['method_name'].lower()+'/'+args['input_data_stats']['parameterization']
	if args['supervised']:
		output_name = output_name+'/sup'
	else:
		output_name = output_name+'/unsup'
	output_name = utils.output_dir(output_name)

	unit = 'gru'
	if args['lstm']:
		unit = 'lstm'

	opt_name = __format_file_name(args['optimizer'])
	return'%s/%s_t%d_l%d_u%d_loss-%s_opt-%s_%s_%d.%s'%(
		output_name, unit, args['timesteps'], args['latent_dim'], args['unit_timesteps'],
		args['loss_func'], opt_name, args['normalization_method'], time.time(), ext)

def get_parse(mode):
	ap = argparse.ArgumentParser()
	ap.add_argument('-m', '--method_name', required=True, help='Method name', choices=METHOD_LIST)

	ap.add_argument('-id', '--input_data', required=False, help='Input data directory', default='../data/h3.6m/euler')
	# ap.add_argument('-od', '--output_data', required=False, help='Output data directory')
	ap.add_argument('-gs', '--generator_size', required=False, help='Size of the batch in the random data generator', default=10000, type=int)
	ap.add_argument('-ts', '--test_size', required=False, help='Size of the test bath', default=100, type=int)
	ap.add_argument('-es', '--embedding_size', required=False, help='Size of the embedding for testing', default=1000, type=int)
	whmtd_list = ['norm_pi', 'norm_std', 'norm_max', 'none']
	ap.add_argument('-w', '--normalization_method', required=False, help='Normalization method.', default='norm_pi', choices=whmtd_list)

	ap.add_argument('-lp', '--load_path', required=False, help='Model path')
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path')
	ap.add_argument('-log', '--log_path', required=False, help='Log file for loss history')

	ap.add_argument('-t', '--timesteps', required=False, help='Total timestep size', default=40, type=int)
	# ap.add_argument('-ti', '--timesteps_in', required=False, help='Input timesteps', default=30, type=int)
	ap.add_argument('-to', '--timesteps_out', required=False, help='Number of output frames (so input size = total timsteps - ouput size)', default=10, type=int)
	ap.add_argument('-ut', '--unit_timesteps', required=False, help='Number of timesteps encoded at the first level', default=10, type=int)
	ap.add_argument('-hs', '--hierarchies', required=False, help='Only encode for these length indices', nargs = '*')
	ap.add_argument('-iter', '--iterations', required=False, help='Number of iterations for training', default=int(1e5), type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default=64, type=int)
	ap.add_argument('-exp', '--expand_time_bound', required=False, help='Expantion of time modality upper bound (only for VL_RNN)', default=10, type=int)
	ap.add_argument('-ld', '--latent_dim', required=False, help='Embedding size', default=800, type=int)
	ap.add_argument('-loss', '--loss_func', required=False, help='Loss function name', default='mean_absolute_error')
	ap.add_argument('-opt', '--optimizer', required=False, help='Optimizer and parameters (use classes in Keras.optimizers)', default='Nadam(lr=0.001)')
	ap.add_argument('-lstm', '--lstm', action='store_true', help='Using LSTM instead of the default GRU')

	ap.add_argument('-sup', '--supervised', action='store_true', help='With action names')
	ap.add_argument('-rep', '--repeat_last', action='store_true', help='Repeat the last frame instead of setting to 0')
	# ap.add_argument('-label', '--only_label', required=False, help='Only load data with this label', nargs = '*')

	ap.add_argument('-debug', '--debug', action='store_true', help='Debug mode (no output file to disk)')

	args = vars(ap.parse_args())

	args['timesteps_in'] = args['timesteps'] - args['timesteps_out']
	assert args['timesteps']  > 0
	args['optimizer'] = 'optimizers.'+args['optimizer']

	# load some statistics and other information about the data
	data_types = ['input_data']
	if 'output_data' in args and args['output_data'] is not None:
		data_types = ['input_data', 'output_data']
	for t in data_types:
		stats = __get_stats(args[t])
		ts = t+'_stats'
		args[ts] = {
			'dim_to_use': stats['dim_to_use'],
			'data_mean': np.array(stats['data_mean']),
			'data_dim': len(stats['dim_to_use']),
			'orig_data_dim': len(stats['data_mean']),
			'parameterization': os.path.basename(args[t])
		}
		for k in ['data_std', 'data_min', 'data_max']:
			args[ts][k] = np.array(stats[k])[stats['dim_to_use']]

		if args['supervised']:
			args['actions'] = stats['action_list']

	# make output path
	if not args['debug']:
		if args[ 'save_path'] is None:
			args['save_path'] = __get_model_path_name(args, 'save')
		if args['log_path'] is None:
			args['log_path'] = __get_model_path_name(args, 'log')

	# load and output data
	if mode == 'train':
		# TODO: add output_data
		return (args,
			__data_generator_random(args['input_data']+'/train/',
				args['input_data_stats'], args, args['generator_size']),
			__data_generator_random(args['input_data']+'/test/',
				args['input_data_stats'], args, args['test_size']),
			__load_validation_data(args['input_data'],
				args['input_data_stats'], args))

	assert(args['load_path'] is not None)
	return (args,
		__data_generator_random(args['input_data']+'/train/',
			args['input_data_stats'], args, args['generator_size']),
		__data_generator(args['input_data']+'/test/',
			args['input_data_stats'], args),
		__load_validation_data(args['input_data'],
				args['input_data_stats'], args))


if __name__ == '__main__':
	# unit test
	for mode in ['train', 'test']:
		args, train_iter, test_iter, valid_iter = get_parse(mode)
		print args
		for x in train_iter:
			print 'train shape', x.shape
			break
		for x in test_iter:
			print 'test shape', x.shape
			break
		for x in valid_iter:
			print 'valid shape', x.shape
			break
