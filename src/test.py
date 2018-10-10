import matplotlib
matplotlib.use('Agg')

import csv
import json
import operator
from itertools import tee

import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models.embedding import viz_embedding, pattern_matching, embedding_utils
from models.format_data import formatter
import viz_poses

def __load_embeddding(model, data_iter, format_func, args, **kwargs):
	stats = args['input_data_stats']
	for x in data_iter:
		x = utils.normalize(x, stats, args['normalization_method'])
		x, _ = format_func(x)
		model.load_embedding(x, **kwargs)
		if args['random_embedding']:
			break # this gives a random sample of the embedding with the data_iter size

def __interpolate(model, stats, args):
	# get 2 random vectors in embedding
	sample_set = model.embedding[model.timesteps-1]
	zs = sample_set[np.random.choice(sample_set.shape[0], 2, replace=False)]
	# interpolate
	itp = embedding_utils.interpolate(zs[0], zs[1])
	# decode everything
	itp = np.concatenate([zs[:1], itp], axis=0)
	itp = np.concatenate([itp, zs[1:]], axis=0)
	x = model.decode(itp)[:,range(0,model.timesteps,5)]
	# visualize
	titles = ['']*itp.shape[0]
	titles[0] = 'start'
	titles[-1] = 'end'
	kwargs = {'row_titles':titles,
		'parameterization':stats['parameterization'],
		'title':'Interpolation'
	}
	viz_poses.plot_batch(x, stats, args, **kwargs)

def __print_scores(scores, supervised):
	actions = scores[scores.keys()[0]].keys()
	for mode in scores:
		print '--------%s--------'%(mode)
		s = {a:scores[mode][a]['y'] for a in scores[mode]}
		utils.print_score(s, 'motion', actions)
		if supervised:
			s = {a:scores[mode][a]['name'] for a in scores[mode]}
			print utils.print_score(s, 'name', actions, False, precision='.4')
		print '======='
		print 'z', np.mean([scores[mode][a]['z'] for a in actions])
		print ''

def __compare_pattern_matching(x_valid_norm, y_valid, model, modalities, args):
	partial_key, complete_key, partial_x_idx, complete_y_idx = modalities

	# encode input
	z_ref = model.encode(x_valid_norm, modality=partial_x_idx)

	# get matches
	z_pred = {}
	for i, mode, z_matched in pattern_matching.batch_all_match(model, z_ref, (partial_key, complete_key)):
		if mode not in z_pred:
			z_pred[mode] = {'z': np.zeros(z_ref.shape)}
		z_pred[mode]['z'][i] = z_matched

	# evaluation
	N = 8 # 8 samples per action
	stats = args['input_data_stats']
	y_valid_norm = utils.normalize(y_valid, stats, args['normalization_method'])
	z_gt = model.encode(y_valid, modality=complete_y_idx)
	scores = {}
	pred_n = complete_y_idx - partial_x_idx
	if pred_n == 0:
		pred_n = complete_y_idx

	for mode, matched in z_pred.iteritems():
		z = matched['z']
		y_pred = model.decode(z)
		z_pred[mode]['y'] = y_pred
		y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
		# iterate action:
		scores[mode] = {}
		for action, i in args['actions'].iteritems():
			s,e = i*8, (i+1)*8
			scores[mode][action] = {
				'z': utils.l2_error(z[s:e], z_gt[s:e]), # compare representation
				'y': (utils.prediction_error(y_pred[s:e],
						y_valid[s:e], stats)[-pred_n:]).tolist()  # compare motion
				}
			# compare action name
			if args['supervised']:
				scores[mode][action]['name'] = utils.classification_error(
					y_pred[s:e], y_valid[s:e], stats)[-pred_n:]

	for mode in z_pred:
		z_pred[mode]['y'] = z_pred[mode]['y'][:,-pred_n:].tolist()
		z_pred[mode]['z'] = z_pred[mode]['z'][:,-pred_n:].tolist()
	z_pred['y_gt'] = y_valid_norm[:,-pred_n:].tolist()

	__print_scores(scores, args['supervised'])

	output = {'scores': scores, 'predictions':z_pred}
	# saving the output
	if not args['debug']:
			filename = args['output_dir']+'__compare_pattern_matching.json'
			json.dump(output, open(filename, 'w'))
			print 'Saved to ', filename

	return output

def __compare_pattern_matching_for_generation(model, stats, args):
	# create input for each action
	x = np.zeros((model.name_dim, model.timesteps, model.input_dim))
	for i in range(model.name_dim):
		x[i,:,i-model.name_dim] = 1
	action_order = sorted(args['actions'].items(), key=operator.itemgetter(1))

	# encode input
	z_ref = model.encode(x, modality=model.timesteps-1)
	# find match
	matches = {'both':{}, 'motion':{}}
	for key in matches.keys():
		modalities = ['name', key]
		for i, mode, z_matched in pattern_matching.batch_all_match(model, z_ref, modalities):
			if mode not in matches[key]:
				matches[key][mode] = {'z': np.zeros(z_ref.shape)}
			matches[key][mode]['z'][i] = z_matched

	for key in matches.keys():
		for mode in matches[key].keys():
			matches[key][mode]['y'] = model.decode(matches[key][mode]['z']).tolist()
			matches[key][mode]['z'] = matches[key][mode]['z'].tolist()

	# visualize
	modes = matches[matches.keys()[0]].keys()
	for a,i in args['actions'].iteritems():
		for key in matches.keys():
			x = np.array([matches[key][m]['y'][i] for m in modes])[:,range(0,model.timesteps,5)]
			kwargs = {'row_titles':modes,
				'parameterization':stats['parameterization'],
				'title':'%s - %s'%(a,key)
			}
			viz_poses.plot_batch(x, stats, args, **kwargs)

	if not args['debug']:
		filename = args['output_dir']+'__compare_pattern_matching.json'
		json.dump(matches, open(filename, 'w'))
		print 'Saved to ', filename


if __name__ == '__main__':
	args, data_iter, test_iter, valid_data = parser.get_parse('test')
	output_dir = '.'.join(args['load_path'].split('.')[:-1])
	args['output_dir'] = output_dir
	args['quantitative_evaluation'] = True
	stats = args['input_data_stats']

	# import model class
	module = __import__('models.'+ args['method_name'])
	method_class = getattr(getattr(module, args['method_name']), args['method_name'])

	model = method_class(args)
	print 'Load model ...'
	model.load(args['load_path'])

	print 'Load embedding ...'
	data_iter, data_iter_ = tee(data_iter)
	__load_embeddding(model, data_iter_, model.format_data, args)

	print 'Computing PCA ...'
	# viz_embedding.plot_convex_hall(model.embedding, args)

	print 'Test interpolaton ...'
	#__interpolate(model, stats, args)

	print 'Comparing pattern matching methods for prediction'
	x_valid, y_valid = model.format_data(valid_data, for_validation=True)
	# x_valid has model.timesteps_in frames
	# y_valid has model.timesteps frames

	x_valid_norm = utils.normalize(x_valid, stats, args['normalization_method'])
	modalities = (model.timesteps_in-1, model.timesteps-1,
				  model.timesteps_in-1, model.timesteps-1)
	#__compare_pattern_matching(x_valid_norm, y_valid, model, modalities, args)

	if args['supervised']:
		# format validation data
		x_valid_no_name = formatter.without_name(model, x_valid_norm)
		y_valid_norm = utils.normalize(y_valid, stats, args['normalization_method'])
		y_valid_no_name = formatter.without_name(model, y_valid_norm)

		print 'Compare pattern matching methods for prediction without name'
		args['output_dir'] = output_dir + '_(no-name)'
		#__compare_pattern_matching(x_valid_no_name, y_valid, model, modalities, args)

		# load different embedding
		data_iter, data_iter_ = tee(data_iter)
		kwargs = {'modalities': formatter.EXPAND_NAMES_MODALITIES}
		__load_embeddding(model, data_iter_, lambda x: formatter.expand_names(model, x), args, **kwargs)

		print 'Compare pattern matching methods for classification'
		args['output_dir'] = output_dir + '_(classification)'
		modalities = ('motion', 'name', model.timesteps-1, model.timesteps-1)
		#__compare_pattern_matching(y_valid_no_name, y_valid, model, modalities, args)

		args['output_dir'] = output_dir + '_(classification-to-both)'
		modalities = ('motion', 'both', model.timesteps-1, model.timesteps-1)
		#__compare_pattern_matching(y_valid_no_name, y_valid, model, modalities, args)

		print 'Compare pattern matchin methods for generation'
		args['output_dir'] = output_dir + '_(generation)'
		__compare_pattern_matching_for_generation(model, stats, args)


