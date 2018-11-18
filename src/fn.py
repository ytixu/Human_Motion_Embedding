import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn import cross_validation
import csv
import h5py

from utils import parser, utils
from models.fn import FN
import viz_2d_poses as viz_poses

LOSS = 10000
CV_SPLIT = 0.2
FAIL_COUNT = 0
SUBACT = 0

martinez_samples = '../../human-motion-prediction/samples.h5'

def get_embedding(x, model, args, stats):
	x = utils.normalize(x, stats, args['normalization_method'])
	y = None

	if args['do_prediction']:
		new_x = model.encode(x[:,:model.timesteps_in], model.timesteps_in-1)
		new_y = np.zeros(x[:,:model.timesteps_in].shape)
		new_y[:,:model.timesteps_out] = x[:,-model.timesteps_out:]
		y = model.encode(new_y, model.timesteps_out-1)

		#x =x[:,:model.timesteps_in]

		#kwargs = {'row_titles':['']*4,
                #        'parameterization':stats['parameterization'],
                #        'title':'fn-test'
                #}
                #x_ = np.concatenate([x[:2], new_y[:2]], axis=0)
                #viz_poses.plot_batch(x_, stats, args, **kwargs)

		return new_x, y
		e = model.encode(x, [model.timesteps_in-1, model.timesteps-1])
		x, y = e[:,0], e[:,1]

	if args['do_classification']:
		motion = np.copy(x)
		motion[:,:,-model.name_dim:] = 0
		y = model.encode(x, model.timesteps-1)
		x = model.encode(motion, model.timesteps-1)

	return x, y

def get_martinez_poses(model, stats, args):
	data = {}
	t = model.timesteps - model.timesteps_in
	with h5py.File( martinez_samples, 'r' ) as h5f:
		# print h5f['expmap/preds'].keys()
		for action,_ in args['actions'].iteritems():
			data[action] = np.zeros((t, model.input_dim))
			try:
				temp = h5f['expmap/preds/%s_%d'%(action, 0)][:t,stats['dim_to_use']]
				data[action][:,:temp.shape[-1]] = temp
			except:
				data[action] = []

	return data

def __eval_loss(fn_model, l2_train, l2_test, args):
	global LOSS, FAIL_COUNT

	new_loss = (l2_train+l2_test)/2
	if new_loss < LOSS:
		print 'Saved model - ', LOSS
		LOSS = new_loss
		#if not args['debug']:
			#fn_model.model.save_weights(args['save_path'], overwrite=True)
		FAIL_COUNT = 0
		return True
	elif FAIL_COUNT > 3:
		fn_model.decay_learning_rate()
	 	print 'new learning rate', fn_model.lr
		FAIL_COUNT = 0
	else:
		FAIL_COUNT += 1
		print FAIL_COUNT, 'failed'

	return False

def __eval(fn_model, x, y, args, stats):
	x_pred = fn_model.predict(x)
	return utils.l2_error(x_pred, y, averaged=True)


def __eval_decoded(model, fn_model, x, y, diff, args, stats):
	x_pred = fn_model.predict(x)
	x_pred = model.decode(x_pred)
	x_pred = utils.unormalize(x_pred, stats, args['normalization_method'])
	err = None

	x_add = model.decode(x+diff)
	x_add = utils.unormalize(x_add, stats, args['normalization_method'])

	if args['do_prediction']:
		print x_pred[:,model.timesteps_in:].shape
		err = utils.prediction_error(x_pred[:,:model.timesteps_out], y, stats, averaged=False)
		#err = utils.prediction_error(x_pred[:,model.timesteps_in:], y, stats, averaged=False)
		utils.print_prediction_score(err, args['actions'], 'FN')

		err = utils.prediction_error(x_add[:,:model.timesteps_out], y, stats, averaged=False)
		#err = utils.prediction_error(x_add[:,model.timesteps_in:], y, stats, averaged=False)
		utils.print_prediction_score(err, args['actions'], 'ADD')
	elif args['do_classification']:
		err = utils.classification_error(x_pred[:,:,-model.name_dim:], y, stats)
		utils.print_classification_score(err, args['actions'])

	return np.mean(err), x_pred

def __plot_sequence(model, y_true, m_pred, x, diff, std_diff, stats, args):
	for action,k in args['actions'].iteritems():
		if len(m_pred[action]) == 0:
			continue
		print action,k
		idx = k*8+SUBACT
		z_pred = fn_model.predict(x[idx:idx+1])
		z = np.zeros((4, z_pred.shape[-1]))
		z[0] = z_pred[0]
		titles = ['Gr. Truth']*6
		titles[-1] = 'ADD'
		titles[2] = 'Prediction'
		titles[1] = 'Res-Seq2Seq'
		for i in range(1,3):
			titles[2+i] = 'Pertube-%d'%((3-i)*2)
			z[i] = np.random.normal(z_pred[0], std_diff/((3-i)*2))
		z[-1] = x[idx]+diff

		y_pred = model.decode(z)
		y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])

		kwargs = {'row_titles':titles,
			'parameterization':stats['parameterization'],
			'title':'FN' + ' ' + action
		}
		print y_pred[:,model.timesteps_in:].shape
		x_ = np.concatenate([[y_true[idx], m_pred[action]], y_pred[:,model.timesteps_in:]], axis=0) #.reshape((6, y_true.shape[1], -1))
		print x_.shape
		viz_poses.plot_batch(x_[:,::2], stats, args, **kwargs)


def train(model, fn_model, data_iter, test_iter, valid_data, stats, args):
	x_valid, y_valid = None, None

	if args['do_prediction']:
		x_valid, y_valid = model.format_data(valid_data, for_prediction=True)
		#x_valid = x_valid[:,:model.timesteps_in]
		x_valid = utils.normalize(x_valid, stats, args['normalization_method'])
		# xp_valid[:,:,-model.name_dim] = 0
		x_valid = model.encode(x_valid, model.timesteps_in-1)

	if args['do_classification']:
		x_valid, y_valid = model.format_data(valid_data, for_classification=True)
		x_valid = utils.normalize(x_valid, stats, args['normalization_method'])
		x_valid = model.encode(x_valid, model.timesteps-1)

	i = 1

	m_data = get_martinez_poses(model, stats, args)

	for x in data_iter:
		print 'ITER', i
		i += 1

		x,y = get_embedding(x, model, args, stats)
		mean_diff = np.mean(y-x, axis=0)
		std_diff = np.std(y-x, axis=0)
		print np.mean(np.std(std_diff)), np.mean(std_diff)

		# training fn model
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=CV_SPLIT)
		history = fn_model.model.fit(x_train, y_train,
					shuffle=True,
					epochs=fn_model.epochs,
					batch_size=fn_model.batch_size,
					validation_data=(x_test, y_test))


		# -- EVALUATION --

		# training error
		rand_idx = np.random.choice(x.shape[0], args['embedding_size'], replace=False)
		l2_train = __eval(fn_model, x[rand_idx], y[rand_idx], args, stats)

		# test error
		x_test = test_iter.next()
		x_test, y_test = get_embedding(x_test, model, args, stats)
		l2_test = __eval(fn_model, x_test, y_test, args, stats)

		print 'MEAN TRAIN', l2_train
		print 'MEAN TEST', l2_test

		# validation error
		l2_valid, y_pred = __eval_decoded(model, fn_model, x_valid, y_valid, mean_diff, args, stats)

		# saving fn model
		if not args['debug'] and __eval_loss(fn_model, l2_train, l2_test, args):
			if args['do_prediction']:
				#pass
				__plot_sequence(model, y_valid, m_data, x_valid, mean_diff, std_diff, stats, args)

		#if not args['debug']:
		#	with open(args['log_path'], 'a+') as f:
		#	 	spamwriter = csv.writer(f)
		#		spamwriter.writerow([new_loss, l2_train, l2_test, l2_valid])

if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('fn')
	stats = args['input_data_stats']
	args['output_dir'] = '.'.join(args['load_path'].split('.')[:-1])

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	print 'Load model ...'
	model.load(args['load_path'])

	fn_model = FN.Forward_NN(args)

	train(model, fn_model, data_iter, test_iter, valid_data, stats, args)
