import csv
import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from model.embedding import viz_embedding

def __load_embeddding(model, data_iter, args):
	for x for data_iter:
		x, _ = model.format_data(x)
		x = utils.normalize(x, stats, args['normalization_method'])
		model.load_embeddding(x)

def __sample_embedding(model, args):
	n = model.embedding.values[0].shape[0]
	rand_idx = np.random.choice(n, args['embedding_size'], replace=False)
	sub_emb = {k:embedding[k][rand_id] for k in model.embedding}
	return sub_emb

if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('test')

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	__load_embeddding(model, data_iter, args)
	sub_emb = __sample_embedding(model, args)

	# PCA reduce and view modalities
	viz_embedding.plot_convex_hall(sub_emb, args)