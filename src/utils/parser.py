import argparse
import operator
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
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

METHOD_LIST = ['test', 'Seq2Seq', 'C_RNN', 'VL_RNN', 'H_RNN', 'HH_RNN', 'H_Seq2Seq', 'HHH_RNN', 'SH_RNN']
OUR_METHODS = ['H_RNN', 'HH_RNN', 'VL_RNN', 'HHH_RNN', 'SH_RNN']

# Get data and information on the data

def __get_stats(data_dir):
	return json.load(open(data_dir+'/stats.json'))

def __get_action_from_file(f):
	return os.path.basename(f).split('_')[0]

def __name_dim(args):
	l = 0
	if args['supervised']:
		l = len(args['actions'])
	return l

def __data_generator(data_dir, stats, args):
	'''
	Data generator for testing.
	Return only the used_dims + one hot name (if applicable)
	'''
	t = args['timesteps']
	actions = args['actions'].keys()

	for f in glob.glob(data_dir+'*.npy'):
		data = np.load(f)[:,stats['dim_to_use']]
		action_name =  __get_action_from_file(f)
		if action_name in actions:
			n,d = data.shape
			n = n-t+1
			l = __name_dim(args)

			x = np.zeros((n, t, d+l))
			for i in range(n):
				# (txd)
				x[i,:,:d] = data[i:i+t,:]

			if args['supervised']:
				if args['add_noise']:
					x[:,:,-l:] = np.random.normal(0,args['add_noise'],x[:,:,-l:].shape)
				x[:,:,-l+args['actions'][action_name]] = 1

			yield x[:100]

def __data_generator_random(data_dir, stats, args, b):
	'''
	Data generator for training.
	Return only the used_dims + one hot name (if applicable)
	Similar to
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L435
	'''
	t = args['timesteps']
	data = {}
	sample_n = 0
	actions = args['actions'].keys()

	for f in glob.glob(data_dir+'*.npy'):
		action_name =  __get_action_from_file(f)
		if action_name in actions:
			data[sample_n] = (np.load(f)[:,stats['dim_to_use']], action_name)
			sample_n += 1

	conseq_n = 1
	l = __name_dim(args)
	x = np.zeros((b,t,stats['data_dim']+l))

	for i in range(args['epoch']):
		sample_idx = np.random.choice(sample_n, b/conseq_n)

		for j,sample_i in enumerate(sample_idx):
			sub_data, action_name = data[sample_i]
			n,d = sub_data.shape
			n = n - conseq_n - t + 1

			rand_idx = np.random.choice(n)
			for k in range(conseq_n):
				x[j*conseq_n+k,:,:d] = sub_data[rand_idx+k:rand_idx+k+t]

			if args['supervised']:
				s, e = j*conseq_n, (j+1)*conseq_n
				x[s:e,:,-l:] = 0
				if args['add_noise']:
					x[s:e,:,-l:] = np.random.normal(0,args['noise_std'],x[s:e,:,-l:].shape)
				x[s:e,:,-l+args['actions'][action_name]] = 1

		yield x

def __load_validation_data(data_dir, stats, args):
	'''
	Load validation data into one matrix.
	'''
	actions = args['actions'].keys()

	files = glob.glob(data_dir+'/valid/*-cond.npy')
	l = __name_dim(args)
	d = stats['data_dim']
	N = 4 # 4 samples per action type and sub-action sequence
	ti,to = args['timesteps_in'],args['timesteps_out']
	x = np.zeros((len(actions)*2*N,args['timesteps'],d+l)) #TODO: generalize batch size len(actions)*2*N

	for f in files:
		action_name =  __get_action_from_file(f)
		if action_name in actions:
			action_idx = args['actions'][action_name]
			sseq = int(f.split('_')[-1].split('-')[0])-1
			s,e = (2*action_idx+sseq)*N,(2*action_idx+sseq+1)*N
			x[s:e,:ti,:d] = np.load(f)[:,-ti:,stats['dim_to_use']]
			x[s:e,-to:,:d] = np.load(f.replace('cond.npy','gt.npy'))[:,:to,stats['dim_to_use']]
			# x is sorted by action type

			if args['supervised']:
				x[s:e,:,-l+action_idx] = 1
	return x

# Get load and save path

def __get_model_path_name(args, file_type):
	'''
	Format name for the log and model file. Name include:
		directory: 	model name
					parameterization (e.g.: euler)
					supervised
					timesteps
		filename: 	rnn unit (lstm or gru)
					latent_dim
					unit_timesteps
					loss_func
					optimizer
					learning rate
					decay
					normalization_method
					repeat_last
					add_noise
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
	output_name = output_name+'/t%d'%(args['timesteps'])
	output_name = utils.output_dir(output_name)

	unit = 'gru'
	if args['lstm']:
		unit = 'lstm'
	misc = ''
	if args['repeat_last']:
		misc = misc + '_repeat_last'
	if args['add_noise']:
		misc = misc + '_noise_%f'%args['noise_std']

	return'%s/%s_e%d_ut%d_loss-%s_opt-%s_lr%f_d%f_%s%s_%d.%s'%(
		output_name, unit, args['latent_dim'], args['unit_timesteps'],
		args['loss_func'], args['optimizer'], args['learning_rate'],
		args['decay'],args['normalization_method'], misc, time.time(), ext)

def get_parse(mode):
	ap = argparse.ArgumentParser()
	ap.add_argument('-m', '--method_name', required=True, help='Method name', choices=METHOD_LIST)

	ap.add_argument('-D', '--input_data', required=False, help='Input data directory', default='../data/h3.6m/euler')
	# ap.add_argument('-od', '--output_data', required=False, help='Output data directory')
	ap.add_argument('--generator_size', required=False, help='Size of the batch in the random data generator, which is the number of sample per epoch during training', default=10000, type=int)
	ap.add_argument('--test_size', required=False, help='Size of the test bath', default=100, type=int)
	ap.add_argument('--embedding_size', required=False, help='Size of the embedding for testing', default=1000, type=int)
	whmtd_list = ['norm_pi', 'norm_std', 'norm_max', 'none']
	ap.add_argument('-n', '--normalization_method', required=False, help='Normalization method.', default='norm_pi', choices=whmtd_list)

	ap.add_argument('-P', '--load_path', required=False, help='Model path')
	ap.add_argument('--save_path', required=False, help='Model save path')
	ap.add_argument('--log_path', required=False, help='Log file for loss history')

	ap.add_argument('-t', '--timesteps', required=False, help='Total timestep size', default=40, type=int)
	ap.add_argument('-o', '--timesteps_out', required=False, help='Number of output frames (so input size = total timsteps - ouput size)', default=10, type=int)
	ap.add_argument('--unit_timesteps', required=False, help='Number of timesteps encoded at the first level', default=10, type=int)
	ap.add_argument('--hierarchies', required=False, help='Only encode for these length indices', nargs = '*')
	ap.add_argument('--epoch', required=False, help='Number of epoch for training', default=1000, type=int)
	ap.add_argument('--batch_size', required=False, help='Batch size', default=64, type=int)
	ap.add_argument('-e', '--latent_dim', required=False, help='Embedding size', default=800, type=int)
	ap.add_argument('-L', '--loss_func', required=False, help='Loss function name', default='mean_absolute_error')
	ap.add_argument('-O', '--optimizer', required=False, help='Optimizer and parameters (use classes in Keras.optimizers)', default='Adam')
	ap.add_argument('-l', '--learning_rate', required=False, help='The learning rate', default=0.001, type=float)
	ap.add_argument('-d', '--decay', required=False, help='The decay factor of the learning rate (decay=1 is no decay)', default='0.95', type=float)
	ap.add_argument('--decay_after', required=False, help='The number of epoch before decaying', default='10', type=int)
	ap.add_argument('--lstm', action='store_true', help='Using LSTM instead of the default GRU')

	ap.add_argument('-s', '--supervised', action='store_true', help='With action names')
	ap.add_argument('-r', '--repeat_last', action='store_true', help='Repeat the last frame instead of setting to 0')
	ap.add_argument('--action', required=False, help='Only load data with this action name', nargs = '*')
	ap.add_argument('--add_noise',required=False, help='Add noise to the action name, indicates the standard deviation for the noise', default=0.0, type=float)
	ap.add_argument('--random_embedding', action='store_true', help='Take a random embedding size of generator_size (for testing).')
	ap.add_argument('--debug', action='store_true', help='Debug mode (no file saved on disk and view model summary)')
	ap.add_argument('--no_save', action='store_true', help='Skip saving model when training, but save log file')

	ap.add_argument('--do_classification', action='store_true', help='Train for classification (when using fn.py)')
	ap.add_argument('--ignore_global', action='store_true', help='Ignore global features')

	args = vars(ap.parse_args())

	args['timesteps_in'] = args['timesteps'] - args['timesteps_out']
	assert args['timesteps_in']  > 0

	# Noise
	if args['add_noise'] > 0:
		args['noise_std'] = args['add_noise']
		args['add_noise'] = True

	# load some statistics and other information about the data
	data_types = ['input_data']
	if 'output_data' in args and args['output_data'] is not None:
		data_types = ['input_data', 'output_data']
	for t in data_types:
		args[t] = args[t].strip('/')
		stats = __get_stats(args[t])
		ts = t+'_stats'
		args[ts] = {
			'dim_to_use': stats['dim_to_use'],
			'data_mean': np.array(stats['data_mean']),
			'data_dim': len(stats['dim_to_use']),
			'orig_data_dim': len(stats['data_mean']),
			'parameterization': os.path.basename(args[t])
		}

		if args['ignore_global']:
			args[ts]['dim_to_use'] = args[ts]['dim_to_use'][6:]
			args[ts]['data_dim'] = args[ts]['data_dim'] - 6
			args[ts]['ignore_global'] = True
		else:
			args[ts]['ignore_global'] = False

		assert args[ts]['parameterization'] == 'euler' or args['normalization_method'] != 'norm_pi'

		for k in ['data_std', 'data_min', 'data_max']:
			args[ts][k] = np.array(stats[k])[stats['dim_to_use']]

		args['actions'] = stats['action_list']
		if args['action'] is not None:
			sorted_actions = sorted(stats['action_list'].items(), key=operator.itemgetter(1))
			sorted_actions = [a for a,_ in sorted_actions if a in args['action']]
			assert len(sorted_actions) != 0
			args['actions'] = {a:i for i,a in enumerate(sorted_actions)}

	# model can do prediction and classification?
	if mode == 'fn':
		if not args['do_classification']:
			args['do_prediction'] = True
		else:
			args['do_prediction'] = False
	else:
		if args['method_name'] == 'C_RNN':
			args['do_classification'] = True
			args['do_prediction'] = False
		else:
			args['do_prediction'] = True
			if args['supervised'] and args['method_name'] in OUR_METHODS:
				args['do_classification'] = True
			else:
				args['do_classification'] = False

	# Parse optimizer
	args['optimizer'] = 'optimizers.'+args['optimizer']

	# load and output data

	if mode == 'train':
		if not args['debug']:
			# make output path
			if args[ 'save_path'] is None:
				args['save_path'] = __get_model_path_name(args, 'save')
			if args['log_path'] is None:
				args['log_path'] = __get_model_path_name(args, 'log')

	else:
		assert(args['load_path'] is not None)

	if mode == 'fn':
		if not args['debug']:
			args['save_path'] = '../' + args['load_path'].strip('.hdf5')
			args['log_path'] = args['save_path']+'_fn.log'
			args['save_path'] = args['save_path']+'_fn.hdf5'

	if mode in ['fn', 'train']:
		# TODO: add output_data
		return (args,
			__data_generator_random(args['input_data']+'/train/',
				args['input_data_stats'], args, args['generator_size']),
			__data_generator_random(args['input_data']+'/test/',
				args['input_data_stats'], args, args['test_size']),
			__load_validation_data(args['input_data'],
				args['input_data_stats'], args))

	data_iter = None
	if args['random_embedding']:
		data_iter = __data_generator_random(args['input_data']+'/train/',
                        args['input_data_stats'], args, args['generator_size'])
	else:
		data_iter =  __data_generator(args['input_data']+'/train/',
                        args['input_data_stats'], args)

	return (args, data_iter,
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
