import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models import ACGAN_RNN

LOSS = 10000
SAVE_TO_DISK = True

def __eval_loss(model, d_loss, g_loss, args):
	'''
	Save model only if loss improved.
	'''
	global LOSS
	new_loss = np.mean([d_loss, g_loss])
	if new_loss < LOSS:
		LOSS = new_loss
		if SAVE_TO_DISK and not args['no_save']:
			model.model.save_weights(args['save_path'], overwrite=True)
		print 'New loss - ', LOSS
	return new_loss

def __print_model(model):
	model.model.summary()
	model.encoder.summary()
	model.decoder.summary()

def train(gan, data_iter, test_iter, valid_data, args):
	'''
	Training routine
	Args
		gan: the RNN gan
		data_iter: the training data generator
		valid_data: the validation data set (only used in evaluating the performance)
		args: other input arguments
	'''
	stats = args['input_data_stats']
	x_valid, y_valid = gan.format_data(valid_data)
	x_valid = utils.normalize(x_valid, stats, args['normalization_method'])

	batch_n = args['batch_size']
	valid = np.ones(batch_size)
	fake = np.zeros(batch_size)
	fake_labels = np.zeros((batch_size, gan.timesteps, gan.name_dim))

	iter_n = 1
	for x in data_iter:
		print 'ITER', iter_n
		iter_n += 1

		# Train Discriminator
		noise = np.random.normal(0, 1, (batch_n, gan.timesteps, gan.input_dim))
		sampled_labels = np.random.randint(0, gan.name_dim, (batch_n, 1))
		gen_seq = gan.encoder.predict([noise, sampled_labels])

		x, y = gan.format_data(x)
		norm_x = utils.normalize(x, stats, args['normalization_method'])

		d_loss_real = gan.decoder.train_on_batch(x, [valid, y])
		d_loss_fake = gan.decoder.train_on_batch(gen_imgs, [fake, fake_labels])
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# Train Generator
		g_loss = gan.encoder.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

		__eval_loss(gan, d_loss, g_loss, args)

		# classification
		# training data
		rand_idx = np.random.choice(norm_x.shape[0], args['embedding_size'], replace=False)
		y_pred = gan.decoder.predict(norm_x[rand_idx])
		err_train = utils.classification_error(y_pred, y[rand_idx], stats)
		utils.print_classification_score(err_train, args['actions'])

		# test data
		x_test = test_iter.next()
		x_test, y_test = model.format_data(x_test)
		x_test = utils.normalize(x_test, stats, args['normalization_method'])
		y_pred = gan.decoder.predict(x_test)
		err_test = utils.classification_error(y_pred, y_test, stats)
		utils.print_classification_score(err_test, args['actions'])

		# valid data
		y_pred = gan.decoder.predict(x_valid)
		err_valid = utils.classification_error(y_pred, y_valid, stats)
		utils.print_classification_score(err_valid, args['actions'])

		if SAVE_TO_DISK:
			with open(args['log_path'], 'a+') as f:
			 	spamwriter = csv.writer(f)
				spamwriter.writerow([d_loss, g_loss, err_train, err_test, err_valid])

if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('train', 'ACGAN_RNN')

	gan = ACGAN_RNN(args['method_name'])

	if args['debug']:
		__print_model(gan)

	if args['load_path'] != None:
		print 'Load gan ...', args['load_path']
		gan.load(args['load_path'])

	SAVE_TO_DISK = not args['debug']
	train(gan, data_iter, test_iter, valid_data, args)
