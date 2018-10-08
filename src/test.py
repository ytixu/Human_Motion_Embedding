import matplotlib
matplotlib.use('Agg')

import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models.embedding import viz_embedding, pattern_matching

def __load_embeddding(model, data_iter, args):
	stats = args['input_data_stats']
	for x in data_iter:
		x, _ = model.format_data(x)
		x = utils.normalize(x, stats, args['normalization_method'])
		model.load_embedding(x)
		break # this gives a random sample of the embedding with the data_iter size

# def __sample_embedding(model, args):
# 	n = model.embedding.values()[0].shape[0]
# 	rand_idx = np.random.choice(n, args['embedding_size'], replace=False)
# 	sub_emb = {k:model.embedding[k][rand_idx] for k in model.embedding}
# 	return sub_emb

def compare_pattern_match(x_valid, y_valid, sample, args):
	c = model.timesteps-1
	z_ref = model.encode(x_valid, modality=c)
	z_pred = {}

	for i, mode, z_matched = pattern_matching.batch_all_match(model, z_ref):
		if mode not in z_pred:
			z_pred = np.zeros(z_ref.shape)
		z_pred[mode][i] = z_matched

	stats = args['input_data_stats']
	z_gt = model.encode(y_valid, modality=model.timesteps-1)
	score = {}
	N = 8 # 8 samples per action

	for mode, z in z_pred.iteritems():
		y_pred = model.decode(z)
		y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
		# iterate action:
		score[mode] = {}
		for action, i in args['actions'].iteritems():
			s,e = i*8, (i+1)*8
			score[mode][a] = { 'z': utils.l2_error(z[s:e], z_gt[s:e]), # compare representation
				'y': utils.prediction_error(y_pred[s:e], y_valid[s:e], stats)} # compare motion

	return score

if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('test')
	args['output_dir'] = '.'.join(args['load_path'].split('.')[:-1])

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	model.load(args['load_path'])

	print 'Load embedding ...'
	__load_embeddding(model, data_iter, args)
	#sub_emb = __sample_embedding(model, args)

	# 1. PCA reduce and view modalities
	print 'Computing PCA ...'
	viz_embedding.plot_convex_hall(model.embedding, args)

	# 2. Compare pattern matching methods
	x_valid, y_valid = model.format_data(valid_data, for_validation=True)
	x_valid = utils.normalize(x_valid, stats, args['normalization_method'])
	compare_pattern_match(x_valid, y_valid)