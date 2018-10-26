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
	if y_pred.shape[-1] != len(args['actions']): # TODO
		y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return np.mean(utils.l2_error(y_pred[:,:,6:], y[:,:,6:]))

def __eval_pred(model, x, y, args, stats):
	'''
	Evalutate prediction error
	'''
	std = []
	y_pred = model.predict(x, return_std=True)
	if len(y_pred) == 2: # TODO: need better way to detect this
		std, y_pred = y_pred
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return std, utils.prediction_error(y_pred[:,:,6:], y[:,:,6:], stats, averaged=False)

def __eval_class(model, x, y, args, stats):
	'''
	Evaluate classification error
	'''
	std = []
	y_pred = model.classify(x, return_std=True)
	print y_pred[0][0]
	if len(y_pred) == 2: # TODO: need better way to detect this
                std, y_pred = y_pred
		print y_pred[0][0]
	return std, utils.classification_error(y_pred, y, stats)

def __combine_prediction_score(score, actions):
	N = 8
	n = len(actions)
	new_s = {}
	keys = ['']*n
	for a,i in actions.iteritems():
		s,e = i*N,(i+1)*N
		new_s[a] = np.mean(score[s:e], axis=0)
		keys[i] = a
	utils.print_score(new_s, 'ADD', keys)

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
	if args['do_prediction']:
		xp_valid, yp_valid = model.format_data(valid_data, for_prediction=True)
		xp_valid = utils.normalize(xp_valid, stats, args['normalization_method'])
		xp_valid[:,:,-model.name_dim] = 0

	if args['do_classification']:
		xc_valid, yc_valid = model.format_data(valid_data, for_classification=True)
		xc_valid = utils.normalize(xc_valid, stats, args['normalization_method'])

	iter_n = 1
	for x in data_iter:
		print 'Epoch', iter_n
		iter_n += 1

		# -- TRAINING --
		# normalization
		x, y = model.format_data(x)
		norm_x = utils.normalize(x, stats, args['normalization_method'])
		norm_y = y
		if y.shape[-1] != len(args['actions']): # TODO
			norm_y = utils.normalize(y, stats, args['normalization_method'])
		# train
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(norm_x, norm_y, test_size=CV_SPLIT)
		history = model.model.fit(x_train, y_train,
					shuffle=True,
					epochs=1,
					batch_size=args['batch_size'],
					validation_data=(x_test, y_test))

		new_loss = __eval_loss(model, history, args)
		# decay
		if iter_n % args['decay_after'] == 0:
			model.decay_learning_rate()
			print 'New learning rate:', model.lr

		# -- EVALUATION --
		# populate embedding with random training data
		rand_idx = np.random.choice(norm_x.shape[0], args['embedding_size'], replace=False)

		# process test data
		x_test = test_iter.next()
		x_test, y_test = model.format_data(x_test)
		x_test = utils.normalize(x_test, stats, args['normalization_method'])

		# training error
		l2_train = __eval(model, norm_x[rand_idx], y[rand_idx], args, stats)
		# test error
		l2_test = __eval(model, x_test, y_test, args, stats)

		print 'MEAN TRAIN', l2_train
		print 'MEAN TEST', l2_test

		l2_valid, mean_std_pred, std_std_pred, log_valid, mean_std_class, std_std_class = 0,0,0,0,0,0
		# prediction error with validation data
		if args['do_prediction']:
			model.load_embedding(norm_x[rand_idx], pred_only=True, new=True)
			std, l2_valid = __eval_pred(model, xp_valid, yp_valid, args, stats)
			mean_std_pred, std_std_pred = 0, 0
			if len(std) > 0:
				mean_std_pred, std_std_pred = np.mean(std), np.std(std)
				print 'Prediction: MEAN STD, STD STD', mean_std_pred, std_std_pred
			__combine_prediction_score(l2_valid, args['actions'])
			l2_valid = np.mean(l2_valid, axis=0).tolist()
			#print 'SHORT-TERM (80-160-320-400ms)', l2_valid[:,utils.SHORT_TERM_IDX]

		# classification error with validation data
		if args['do_classification']:
			model.load_embedding(norm_x[rand_idx], class_only=True, new=True)
			std, log_valid = __eval_class(model, xc_valid, yc_valid, args, stats)
			if len(std) > 0:
				mean_std_class, std_std_class = np.mean(std), np.std(std)
				print 'Classification: MEAN STD, STD STD', mean_std_class, std_std_class
			utils.print_classification_score(log_valid, args['actions'])
			log_valid = np.mean(log_valid, axis=0).tolist()

		if SAVE_TO_DISK:
			with open(args['log_path'], 'a+') as f:
			 	spamwriter = csv.writer(f)
				spamwriter.writerow([new_loss, l2_train, l2_test, l2_valid, mean_std_pred, std_std_pred, log_valid, mean_std_class, std_std_class])


if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('train')

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	if args['debug']:
		__print_model(model)
	if args['load_path'] != None:
		print 'Load model ...', args['load_path']
		model.load(args['load_path'])

	SAVE_TO_DISK = not args['debug']
	train(model, data_iter, test_iter, valid_data, args)
