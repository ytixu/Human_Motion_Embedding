import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils
# from models.Seq2Seq import Seq2Seq
# from models.VL_RNN import VL_RNN

CV_SPLIT = 0.2
LOSS = 1000

SAVE_TO_DISK = True

def __eval_loss(model, history, args):
	'''
	Save model only if loss improved.
	'''
	global LOSS
	print history.history['loss']
	new_loss = np.mean(history.history['loss'])
	if new_loss < LOSS:
		LOSS = new_loss
		if SAVE_TO_DISK:
			model.model.save_weights(args['save_path'], overwrite=True)
		print 'Saved model - ', LOSS
	return new_loss

def __eval(model, x, y, args, stats):
	'''
	Compare results using l2 distance of the Euler angle.
	TODO: need to adapt to expmap.
	'''
	y_pred = model.autoencode(x)
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return utils.euler_error(y, y_pred, stats)

def __eval_pred(model, x, y, args, stats):
	std, y_pred = model.predict(x, return_std=True)
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return std, utils.euler_error(y, y_pred, stats)

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

		# populate embedding with random training data
		rand_idx = np.random.choice(norm_x.shape[0], args['embedding_size'], replace=False)
		model.load_embedding(norm_x[rand_idx])

		# process test data
		x_test = test_iter.next()
		x_test, y_test = model.format_data(x_test)
		x_test = utils.normalize(x_test, stats, args['normalization_method'])

		# training error
		mse = __eval(model, norm_x[rand_idx], y[rand_idx], args, stats)
		# test error
		mse_test = __eval(model, x_test, y_test, args, stats)
		# prediction error with validation data
		std, mse_valid = __eval_pred(model, norm_x_valid, valid_data, args, stats)
		mean_std, std_std = 0, 0
		if len(std) > 0:
			mean_std, std_std = np.mean(std), np.std(std)
			print 'MEAN STD, STD STD', mean_std, std_std

		print 'MEAN TRAIN', np.mean(mse)
		print 'MEAN TEST', np.mean(mse_test)
		print 'SHORT-TERM (80-160-320-400ms)', utils.list_short_term(model, mse_valid)

		if SAVE_TO_DISK:
			with open(args['log_path'], 'a+') as f:
			 	spamwriter = csv.writer(f)
			 	spamwriter.writerow([new_loss, mse, mse_test, mse_valid, mean_std, std_std])


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
