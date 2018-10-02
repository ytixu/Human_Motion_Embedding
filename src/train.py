import numpy as np
from itertools import tee
from sklearn import cross_validation

from utils import parser
from models import Seq2Seq

CV_SPLIT = 0.2
RAND_EVAL = 500
LOSS = 1000

def __eval_loss(model, history, args):
	global LOSS
	print history.history['loss']
	new_loss = np.mean(history.history['loss'])
	if new_loss < LOSS:
		LOSS = new_loss
		# model.model.save_weights(args['save_path'], overwrite=True)
		print 'Saved model - ', loss

def eval(model, x, y, args, stats):
	y_pred = model.predict(x)
	y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
	return utils.euler_error(y, y_pred, stats)

def train(model, data_iter, valid_data, args):
	stats = args['input_data_stats']
	x_valid, y_valid = model.format_data(valid_data)
	norm_x_valid = utils.normalize(x_valid, stats, args['normalization_method'])

	for x in data_iter:
		# normalization
		norm_x = utils.normalize(x, stats, args['normalization_method'])
		x, y = model.format_data(x)
		norm_x, norm_y = self.format_data(norm_x)
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(norm_x, norm_y, test_size=CV_SPLIT)

		history = self.autoencoder.fit(x_train, y_train,
					shuffle=True,
					epochs=1,
					batch_size=args['batch_size'],
					validation_data=(x_test, y_test))

		rand_idx = np.random.choice(norm_x.shape[0], RAND_EVAL, replace=False)

		mse = self.eval(norm_x[rand_idx], y[rand_idx])
		mse_valid = self.validate(norm_x_valid, y_valid)

		print 'MSE', np.mean(mse)
		print 'MSE VALID', np.mean(mse_valid)
		print 'MSE PRED', mse_valid[utils.SHORT_TERM_IDX] # TODO

		# with open(args['log_path'], 'a+') as f:
		# 	spamwriter = csv.writer(f)
		# 	spamwriter.writerow([new_loss, mse, mse_valid])


if __name__ = '__main__':
	args, data_iter, valid_data = get_parse('train')
	model = (globals()[args['method']])()
	train(model, data_iter, valid_data, args)
