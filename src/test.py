import matplotlib
matplotlib.use('Agg')

import csv
import json
import operator
from itertools import tee

import numpy as np
from sklearn import cross_validation

from utils import parser, utils
from models.utils import viz_embedding, pattern_matching, embedding_utils, formatter
import viz_poses

def __load_embeddding(model, data_iter, args, **kwargs):
	stats = args['input_data_stats']
	for x in data_iter:
		x = utils.normalize(x, stats, args['normalization_method'])
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
	x = utils.unormalize(x, stats, args['normalization_method'])
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

def __compare_pattern_matching(part_data, comp_data, model, modalities, args):
	partial_key, complete_key, partial_x_idx, complete_y_idx = modalities

	# encode input
	z_ref = model.encode(part_data, modality=partial_x_idx)

	# get matches
	z_pred = {}
	for i, mode, z_matched in pattern_matching.batch_all_match(model, z_ref, (partial_key, complete_key)):
		if mode not in z_pred:
			z_pred[mode] = {'z': np.zeros(z_ref.shape)}
		z_pred[mode]['z'][i] = z_matched

	# evaluation
	N = 8 # 8 samples per action
	stats = args['input_data_stats']
	cpl_data_norm = utils.normalize(cpl_data, stats, args['normalization_method'])
	z_gt = model.encode(cpl_data, modality=complete_y_idx)
	scores = {}
	pred_n = complete_y_idx - partial_x_idx
	if pred_n == 0:
		pred_n = complete_y_idx

	for mode, matched in z_pred.iteritems():
		z = matched['z']
		y_pred = model.decode(z)
		y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
		z_pred[mode]['y'] = y_pred
		# iterate action:
		scores[mode] = {}
		for action, i in args['actions'].iteritems():
			s,e = i*N, (i+1)*N
			scores[mode][action] = {
				'z': utils.l2_error(z[s:e], z_gt[s:e]), # compare representation
				'y': (utils.prediction_error(y_pred[s:e],
						cpl_data[s:e], stats)[-pred_n:]).tolist()  # compare motion
				}
			# compare action name
			if args['supervised']:
				scores[mode][action]['name'] = utils.classification_error(
					y_pred[s:e], cpl_data[s:e], stats)[-pred_n:]

	modes = sorted(z_pred.keys())
	print modes
	for mode in z_pred:
		z_pred[mode]['y'] = z_pred[mode]['y'][:,-pred_n:].tolist()
		z_pred[mode]['z'] = z_pred[mode]['z'][:,-pred_n:].tolist()
	z_pred['y_gt'] = cpl_data[:,-pred_n:].tolist()

	# print score
	__print_scores(scores, args['supervised'])

	# visualize
	for action,i in args['actions'].iteritems():
		x = np.concatenate([[z_pred['y_gt'][i*N]], [z_pred[m]['y'][i*N] for m in modes]], axis=0)
		kwargs = {'row_titles':['gt'] + modes,
			'parameterization':stats['parameterization'],
			'title':action
		}
		if x.shape[1] == model.timesteps:
			x = x[:,range(0,model.timesteps,5)]
		viz_poses.plot_batch(x, stats, args, **kwargs)
		if args['debug']: break # just show one

	# saving the output
	output = {'scores': scores, 'predictions':z_pred}

	if not args['debug']:
			filename = args['output_dir']+'.json'
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
			y_pred = model.decode(matches[key][mode]['z'])
			y_pred = utils.unormalize(y_pred, stats, args['normalization_method'])
			matches[key][mode]['y'] = y_pred.tolist()
			matches[key][mode]['z'] = matches[key][mode]['z'].tolist()

	# visualize
	modes = matches[matches.keys()[0]].keys()
	for action,i in args['actions'].iteritems():
		for key in matches.keys():
			x = np.array([matches[key][m]['y'][i] for m in modes])[:,range(0,model.timesteps,5)]
			kwargs = {'row_titles':modes,
				'parameterization':stats['parameterization'],
				'title':'%s - %s'%(action,key)
			}
			viz_poses.plot_batch(x, stats, args, **kwargs)
		if args['debug']: break # just show two

	if not args['debug']:
		filename = args['output_dir']+'.json'
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
	__load_embeddding(model, data_iter_, args)

	print 'Computing PCA ...'
	args['output_dir'] = output_dir + '_PCA'
	viz_embedding.plot_convex_hall(model.embedding, args)

	print 'Test interpolaton ...'
	args['output_dir'] = output_dir + '_interpolation'
	__interpolate(model, stats, args)

	print 'Comparing pattern matching methods for prediction'
	xp_valid,_ = model.format_data(valid_data, for_prediction=True)
	xp_valid = utils.normalize(xp_valid, stats, args['normalization_method'])
	modalities = (model.timesteps_in-1, model.timesteps-1,
				  model.timesteps_in-1, model.timesteps-1)
	args['output_dir'] = output_dir + '_pattern_matching'
	__compare_pattern_matching(xp_valid, valid_data, model, modalities, args)

	if args['supervised']:
		# format validation data
		xc_valid,_ = model.format_data(valid_data, for_classification=True)
		xc_valid = utils.normalize(xc_valid, stats, args['normalization_method'])
		xp_valid_unamed,_ = model.format_data(xc_valid, for_prediction=True)
		names_valid = formatter.without_motion(model, valid_data)

		print 'Compare pattern matching methods for prediction without name'
		args['output_dir'] = output_dir + '_pattren_matching_(no-name)'
		__compare_pattern_matching(xp_valid_unamed, valid_data, model, modalities, args)

		# load different embedding
		# data_iter, data_iter_ = tee(data_iter)
		kwargs = {'class_only': True}
		__load_embeddding(model, data_iter, args, **kwargs)

		print 'Compare pattern matching methods for classification'
		args['output_dir'] = output_dir + '_pattern_matching_(classification)'
		modalities = ('motion', 'name', model.timesteps-1, model.timesteps-1)
		__compare_pattern_matching(xc_valid, names_valid, model, modalities, args)

		args['output_dir'] = output_dir + '_pattern_matching_(classification-to-both)'
		modalities = ('motion', 'both', model.timesteps-1, model.timesteps-1)
		__compare_pattern_matching(xc_valid, data_valid, model, modalities, args)

		print 'Compare pattern matchin methods for generation'
		args['output_dir'] = output_dir + '_pattern_matching_(generation)'
		__compare_pattern_matching_for_generation(model, stats, args)


