import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models.seq2seq import Seq2Seq

CV_SPLIT = 0.2
RAND_EVAL = 500
LOSS = 1000

def __eval_loss(model, history, args):
	'''
	Save model only if loss improved.
	'''
	global LOSS
	print history.history['loss']
	new_loss = np.mean(history.history['loss'])
	if new_loss < LOSS:
		LOSS = new_loss
		model.model.save_weights(args['save_path'], overwrite=True)
		print 'Saved model - ', LOSS
	return new_loss

def __eval(model, x, y, args, stats):
	'''
	Compare results using l2 distance of the Euler angle.
	TODO: need to adapt to expmap.
	'''
	y_pred = model.predict(x)
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return utils.euler_error(y, y_pred, stats)

def train(model, data_iter, valid_data, args):
	'''
	Training routine
	Args
		model: the RNN model
		data_iter: the training data generator
		valid_data: the validation data set (only used in evaluating the performance)
		args: other input arguments
	'''
	stats = args['input_data_stats']
	x_valid, y_valid = model.format_data(valid_data)
	norm_x_valid = utils.normalize(x_valid, stats, args['normalization_method'])

	for x in data_iter:
		# normalization
		norm_x = utils.normalize(x, stats, args['normalization_method'])
		x, y = model.format_data(x)
		norm_x, norm_y = model.format_data(norm_x)
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(norm_x, norm_y, test_size=CV_SPLIT)

		history = model.model.fit(x_train, y_train,
					shuffle=True,
					epochs=1,
					batch_size=args['batch_size'],
					validation_data=(x_test, y_test))

		new_loss = __eval_loss(model, history, args)
		rand_idx = np.random.choice(norm_x.shape[0], RAND_EVAL, replace=False)

		mse = __eval(model, norm_x[rand_idx], y[rand_idx], args, stats)
		mse_valid = __eval(model, norm_x_valid, y_valid, args, stats)

		print 'MEAN TRAIN', np.mean(mse)
		print 'MEAN VALID', np.mean(mse_valid)
		print 'SHORT-TERM (80-160-320-400ms)', mse_valid[utils.SHORT_TERM_IDX] # TODO

		with open(args['log_path'], 'a+') as f:
		 	spamwriter = csv.writer(f)
		 	spamwriter.writerow([new_loss, mse, mse_valid])


if __name__ == '__main__':
	args, data_iter, valid_data = parser.get_parse('train')
	model = (globals()[args['method_name']])(args)
	train(model, data_iter, valid_data, args)
