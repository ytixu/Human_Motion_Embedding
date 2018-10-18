import argparse
import time
import glob
import os.path
import numpy as np

def get_model_load_name(model_name):
	return '../../models/%s_%d.hdf5'%(model_name, time.time())

def get_log_name(model_name):
	return '../../models/%s_%d.log'%(model_name, time.time())

def get_data(data_path):
	data = np.load(glob.glob(data_path)[0])
	return data.shape[-1]

def data_generator(data_path, n):
	files = glob.glob(data_path)
	for i in range(n):
		x = []
		y = []
		rand_f = np.random.choice(len(files), 10, replace=False)
		for f in range(len(files)):
			print f
			data = np.load(files[f])
			N = data.shape[0]
			# rand1 = np.random.choice(N, 3000, replace=False)
			rand2 = np.random.choice(N, 3000, replace=False)
			# new_x = np.concatenate([data[rand1,0],data[rand2,1]], axis=0)
			# new_y = np.concatenate([data[rand1,2],data[rand2,2]], axis=0)
			new_x = data[rand2,0]
			new_y = data[rand2,1]
			if len(x) == 0:
				x = new_x
				y = new_y
			else:
				x = np.concatenate([x,new_x], axis=0)
				y = np.concatenate([y, new_y], axis=0)

		print x.shape, y.shape
		yield x, y


def get_parse(model_name):
	ap = argparse.ArgumentParser()
	list_of_modes = ['train', 'sample']
	ap.add_argument('-id', '--data_path', required=True, help='Input data directory')
	ap.add_argument('-m', '--mode', required=False, help='Choose between training mode or sampling mode.', default='train', choices=list_of_modes)
	ap.add_argument('-ep', '--epochs', required=False, help='Number of epochs', default='3', type=int)
	ap.add_argument('-bs', '--batch_size', required=False, help='Batch size', default='16', type=int)
	ap.add_argument('-sp', '--save_path', required=False, help='Model save path', default=get_model_load_name(model_name))
	ap.add_argument('-p', '--periods', required=False, help='Number of iterations of the data', default='100', type=int)

	args = vars(ap.parse_args())
	dim = get_data(args['data_path']+'*')
	data_iterator = data_generator(args['data_path']+'*', args['periods'])
	args['input_dim'], args['output_dim'] = dim, dim

	return data_iterator, args
