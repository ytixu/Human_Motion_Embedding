import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models.fn import FN

LOSS = 10000
CV_SPLIT = 0.2


def get_embedding(x, model, stats, args):
	if args['do_prediction']:
		return model.encode(x, model.timesteps_in-1)
	elif args['do_classification']:
		return model.encode(x, model.timesteps-1)

def __eval_loss(fn_model, l2_train, l2_test, args):
	new_loss = (l2_train+l2_test)/2
	if new_loss < LOSS:
		print 'Saved model - ', LOSS
		LOSS = new_loss
		if not args['debug']:
			fn_model.model.save_weights(args['save_path'], overwrite=True)
		count = 0
	elif count > 3:
		fn_model.decay_learning_rate()
	 	print 'new learning rate', fn_model.lr
		count = 0
	else:
		count += 3
		print count, 'failed'

def __eval(fn_model, x, y, args, stats):
	x_pred = fn_model.predict(x)
	return utils.l2_error(x_pred, y, average=True)


def __eval_decoded(model, fn_model, x, y, args, stats):
	x_pred = fn_model.predict(x)
	x_pred = model.decode(x_pred)
	x_pred =utils.unnormalize(x_pred, stats, args['normalization_method'])
	err = utils.l2_error(x_pred, y)

	if args['do_prediction']:
		utils.print_prediction_score(err, args['actions'])
	elif args['do_classification']:
		utils.print_classification_score(err, args['actions'])

	return np.mean(err)


def train(model, fn_model, data_iter, test_iter, valid_data, stats, args):
	global LOSS

	if args['do_prediction']:
		xp_valid, yp_valid = model.format_data(valid_data, for_prediction=True)
		xp_valid = utils.normalize(xp_valid, stats, args['normalization_method'])
		# xp_valid[:,:,-model.name_dim] = 0
		xp_valid = model.encode(xp_valid, model.timesteps_in-1)

	if args['do_classification']:
		xc_valid, yc_valid = model.format_data(valid_data, for_classification=True)
		xc_valid = utils.normalize(xc_valid, stats, args['normalization_method'])
		xc_valid = model.encode(xc_valid, model.timesteps-1)

	i = 1

	for x in data_iter:
		print 'ITER', i
		i += 1
		count = 0

		# get embedding
		x = utils.normalize(x, stats, args['normalization_method'])
		y = None

		if args['do_prediction']:
			e = model.encode(x, [model.timesteps_in-1, model.timesteps-1])
			x, y = e[:,0], e[:,1]

		if args['do_classification']:
			x = model.encode(name, model.timesteps-1)
			name = np.copy(x)
			name = x[:,:-model.name_dim] = 0
			y = model.encode(name, model.timesteps-1)

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
		x_test, y_test = model.format_data(x_test)
		x_test = utils.normalize(x_test, stats, args['normalization_method'])
		x_test = get_embedding(x_test)
		y_test = get_embedding(y_test)
		l2_test = __eval(fn_model, x_test, y_test, args, stats)

		print 'MEAN TRAIN', l2_train
		print 'MEAN TEST', l2_test

		# validation error
		l2_valid = __eval_decoded(model, fn_model, xp_valid, yp_valid, args, stats)

		# saving fn model
		__eval_loss(fn_model, l2_train, l2_test, args)

		if not args['debug']:
			with open(args['log_path'], 'a+') as f:
			 	spamwriter = csv.writer(f)
				spamwriter.writerow([new_loss, l2_train, l2_test, l2_valid])

if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('fn')
	stats = args['input_data_stats']

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	print 'Load model ...'
	model.load(args['load_path'])

	fn_model = FN.Forward_NN(args)

	train(model, fn_model, data_iter, test_iter, valid_data, stats, args)
