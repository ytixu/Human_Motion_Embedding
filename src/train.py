import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils

CV_SPLIT = 0.2
LOSS = 1000

SAVE_TO_DISK = True

def __eval_loss(model, history, args):
	'''
	Save model only if loss improved.
	'''
	global LOSS
	new_loss = np.mean(history.history['loss'])
	if new_loss < LOSS:
		LOSS = new_loss
		if SAVE_TO_DISK and not args['no_save']:
			model.model.save_weights(args['save_path'], overwrite=True)
		print 'New loss - ', LOSS
	return new_loss

def __eval(model, x, y, args, stats):
	'''
	Evaluate training error using l2 distance of the Euler angle.
	'''
	y_pred = model.autoencode(x)
	#y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return np.mean(utils.l2_error(y_pred, y))

def __eval_pred(model, x, y, args, stats):
	'''
	Evalutate prediction error
	'''
	std = []
	y_pred = model.predict(x, return_std=True)
	if len(y_pred) == 2: # TODO: need better way to detect this
		std, y_pred = y_pred
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	print y_pred.shape, y.shape
	return std, utils.prediction_error(y_pred, y, stats)

def __eval_class(model, x, y, args, stats):
	'''
	Evaluate classification error
	'''
	return  utils.classification_error(model.classify(x, return_std=True), y, stats)

	#std, y_pred = model.classify(x, return_std=True)
	return std, np.mean(utils.classification_error(y_pred, y, stats))

def __print_model(model):
	model.model.summary()
	model.encoder.summary()
	model.decoder.summary()

def train(model, data_iter, test_iter, valid_data, args):
	'''
	Training routine
	Args
		model: the RNN model
		data_iter: the training data generator
		valid_data: the validation data set (only used in evaluating the performance)
		args: other input arguments
	'''
	stats = args['input_data_stats']
	xp_valid, yp_valid = model.format_data(valid_data, for_prediction=True)
	xp_valid = utils.normalize(xp_valid, stats, args['normalization_method'])
	if args['supervised']:
		xc_valid, yc_valid = model.format_data(valid_data, for_classification=True)
		xc_valid = utils.normalize(xc_valid, stats, args['normalization_method'])

	iter = 1
	for x in data_iter:
		print 'ITER', iter
		iter += 1

		# normalization
		x, y = model.format_data(x)
		norm_x = utils.normalize(x, stats, args['normalization_method'])
		# norm_y = utils.normalize(y, stats, args['normalization_method'])
		norm_y = y
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(norm_x, norm_y, test_size=CV_SPLIT)
		history = model.model.fit(x_train, y_train,
					shuffle=True,
					epochs=1,
					batch_size=args['batch_size'],
					validation_data=(x_test, y_test))

		new_loss = __eval_loss(model, history, args)

		# populate embedding with random training data
		rand_idx = np.random.choice(norm_x.shape[0], args['embedding_size'], replace=False)
		model.load_embedding(norm_x[rand_idx], pred_only=True, new=True)

		# process test data
		x_test = test_iter.next()
		x_test, y_test = model.format_data(x_test)
		x_test = utils.normalize(x_test, stats, args['normalization_method'])

		# training error
		l2_train = __eval(model, norm_x[rand_idx], y[rand_idx], args, stats)
		# test error
		l2_test = __eval(model, x_test, y_test, args, stats)

		# prediction error with validation data
		# std, l2_valid = __eval_pred(model, xp_valid, yp_valid, args, stats)
		# mean_std_pred, std_std_pred = 0, 0
		# if len(std) > 0:
		# 	mean_std_pred, std_std_pred = np.mean(std), np.std(std)
		# 	print 'Prediction: MEAN STD, STD STD', mean_std_pred, std_std_pred

		# classification error with validation data
		if args['supervised']:
			# TODO: need to fix this for randomly expanded names
			print yc_valid.shape
			model.load_embedding(norm_x[rand_idx], class_only=True, new=True)
			log_valid = __eval_class(model, xc_valid, yc_valid, args, stats)
			# std, log_valid = __eval_class(model, xc_valid, yc_valid, args, stats)
			# mean_std_class, std_std_class = np.mean(std), np.std(std)
			# print 'Classification: MEAN STD, STD STD', mean_std_class, std_std_class

		print 'MEAN TRAIN', l2_train
		print 'MEAN TEST', l2_test
		# print 'SHORT-TERM (80-160-320-400ms)', l2_valid[utils.SHORT_TERM_IDX]
		if args['supervised']:
			print 'CLASSIFICATION'
			utils.print_classification_score(log_valid, args['actions'])

		if SAVE_TO_DISK:
			with open(args['log_path'], 'a+') as f:
			 	spamwriter = csv.writer(f)
				if ['supervised']:
					spamwriter.writerow([new_loss, l2_train, l2_test, l2_valid, mean_std_pred, std_std_pred, log_valid, mean_std_class, std_std_class])
				else:
				 	spamwriter.writerow([new_loss, l2_train, l2_test, l2_valid, mean_std_pred, std_std_pred])


if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('train')

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	if args['debug']:
		__print_model(model)

	SAVE_TO_DISK = not args['debug']
	train(model, data_iter, test_iter, valid_data, args)
