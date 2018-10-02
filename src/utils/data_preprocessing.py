import argparse
import csv
import glob
import json

import numpy as np
from tqdm import tqdm

import converter
import utils

DATA_DIR = '../../data/h3.6m/'

# ids copied from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L677
TRAIN_SUBJECT_ID = [1,6,7,8,9,11]
TEST_SUBJECT_ID = [5]
# actions copied from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L637
ACTIONS = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

def readCSVasFloat(filename, action=None, subact=None):
	'''
	Modified from:
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L195
	Unused input values action, subact
	"""
	Borrowed from SRNN code. Reads a csv and returns a float matrix.
	https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
	-------
	Args
		filename: string. Path to the csv file
	Yield
		returnArray: the read data in a float32 matrix
	"""
	'''
	returnArray = []
	lines = open(filename).readlines()
	for line in lines:
		line = line.strip().split(',')
		if len(line) > 0:
			returnArray.append(np.array([np.float32(x) for x in line]))

	returnArray = np.array(returnArray)
	yield returnArray

def find_indices_srnn( action, subj ):
    '''
    Hard copy of the indices as produced in:
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """
    '''
    return {'walking' : [[1087, 1145, 660, 201],[955, 332, 304, 54]],
    'eating' : [[1426, 1087, 1329, 1145],[374, 156, 955, 332]],
    'smoking' : [[1426, 1087, 1329, 1145],[1398, 1180, 955, 332]],
    'discussion' : [[1426, 1398, 1180, 332],[2063, 1087, 1145, 1438]],
    'directions' : [[1426, 1087, 1145, 1438],[374, 156, 332, 665]],
    'greeting' : [[402, 63, 305, 121],[1398, 1180, 955, 332]],
    'phoning' : [[1426, 1087, 1329, 332],[374, 156, 121, 414]],
    'posing' : [[402, 63, 835, 955],[374, 156, 305, 121]],
    'purchases' : [[1087, 955, 332, 304],[1180, 1145, 660, 201]],
    'sitting' : [[1426, 1087, 1329, 1145],[1398, 1180, 955, 332]],
    'sittingdown' : [[1426, 1087, 1145, 1438],[1398, 1180, 332, 1689]],
    'takingphoto' : [[1426, 1180, 1145, 1438],[1087, 955, 332, 660]],
    'waiting' : [[1426, 1398, 1180, 332],[2063, 1087, 1145, 1438]],
    'walkingdog' : [[402, 63, 305, 332],[374, 156, 121, 414]],
    'walkingtogether' : [[1087, 1329, 1145, 660],[1180, 955, 332, 304]]}[action][subj-1]

def readCSVasFloat_for_validation(filename, action, subact):
	'''
	Stitched from:
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L195
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L216
	'''
	with open(filename, 'r') as csvfile:
		lines = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))

		# (from una-dinosauria/human-motion-prediction.seq2seq_model.get_batch_srnn)
		frames = find_indices_srnn( action, subact )

		# 150 frames (as in una-dinosauria/human-motion-prediction.seq2seq_model.get_batch_srnn)
		# 50 for the conditioned sequence when duing motion prediction
		for i, idx in enumerate(frames):
			# we skipped every second frame
			yield lines[2*idx:2*(idx+150)]

def load_data(path_to_dataset, subjects, actions, func=readCSVasFloat):
	'''
	Modified from:
		https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L216
	"""
	Borrowed from SRNN code. This is how the SRNN code reads the provided .txt files
	https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270
	-------
	Args
		path_to_dataset: string. directory where the data resides
		subjects: list of numbers. The subjects to load
		actions: list of string. The actions to load
		func: function to read the data file
	Yield
		k=(subject, action, subaction, 'even'), v=(nxd) data
 	"""
 	'''
	for subj in subjects:
		for action_idx in np.arange(len(actions)):

			action = actions[ action_idx ]

			for subact in [1, 2]:  # subactions
				print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

				filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)

				for action_sequence in func(filename, action, subact):
					n, _ = action_sequence.shape
					even_list = range(0, n, 2)
					yield (subj, action, subact), action_sequence[even_list, :]

def load_validation(path_to_dataset, actions):
	'''
	Similar to load_data()
	-------
	Args
		path_to_dataset: string. directory where the data resides
		actions: list of string. The actions to load
	Yield
		k=(subject, action, subaction, 'even'), s1=50 conditioned sequence, s2=100 ground truth sequence
	'''
	for k, data_sequences in load_data(path_to_dataset, TEST_SUBJECT_ID, actions, readCSVasFloat_for_validation):
		yield k, data_sequences[:50], data_sequences[50:]

def convert(to_type='euler', vis=False):
	'''
	Converting and saving the data in euler, euclidean or exmponential map
	'''
  	directory = DATA_DIR+to_type
  	from_directory = DATA_DIR+'dataset'

	def convert_to_type(x):
		if to_type == 'expmap':
			if vis:
				converter.animate(converter.sequence_expmap2xyz(x))
		elif to_type == 'euler':
			x = converter.sequence_expmap2euler(x)
			if vis:
				converter.animate(converter.sequence_euler2xyz(x))
		else:
			x = converter.sequence_expmap2xyz(x)
			if vis:
				converter.animate(x)
		return x

  	if not vis:
	  	utils.create_dir(directory+'/train/')
	  	utils.create_dir(directory+'/test/')
  		utils.create_dir(directory+'/valid/')

	#training set and test set
	train_set = load_data(from_directory, TRAIN_SUBJECT_ID, ACTIONS)
  	test_set = load_data(from_directory, TEST_SUBJECT_ID,  ACTIONS)
	for data_name, data_set in [('train',train_set), ('test',test_set)]:
		for k, x in data_set:
			print 'save', k, x.shape
			subject, action, subact = k
			data = convert_to_type(x)
			print data.shape
			if not vis:
				np.save(directory+'/%s/%s_%d_%d.npy'%(data_name, action, subject, subact), data)

	#validation for motion prediction
	valid_set = load_validation(from_directory,  ACTIONS)
  	for k, cond, gt in valid_set:
  		subject, action, subact = k
  		for data_name, x in [('cond',cond), ('gt',gt)]:
	  		data = convert_to_type(x)
			print data.shape
			if not vis:
				np.save(directory+'/valid/%s_%d_%d-%s.npy'%(action, subject, subact, data_name), data)

		# This is for euler, you can change it to other parameterization
		# Uncomment this to visualize the 2 sequence together
		# converter.animate(converter.sequence_euler2xyz(np.concatenate([cond, gt], axis=0)))


def __get_data(files):
	'''
	Load and concatenate all .npy files.
	'''
	data = []
	for i,f in enumerate(tqdm(files)):
		if len(data) == 0:
			data = np.load(f)
		else:
			data = np.concatenate([np.load(f), data], axis = 0)
	return data

def whitening(to_type='euler'):
	'''
	Similar to https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L302
	"""
	Save statistics for whitening the data.
	Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
	https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
	"""
	'''

	directory = DATA_DIR+to_type+'/train/'

	files = glob.glob(directory+'*.npy')
	data = __get_data(files)

	data_mean = np.mean(data, axis=0)
	data_std = np.std(data, axis=0)

	'''
	For euler:
	We are ignoring all joints with no variance when computing the l2 error. Same as in
	https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L604
	and https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L221
	This is Martinez et al.'s comment:
	# Now compute the l2 error. The following is numpy port of the error
	# function provided by Ashesh Jain (in matlab), available at
	# https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
	'''
	# First 6 values are global rotation and translation, which are also ignored.
	dimensions_to_ignore = range(6)+list(np.where(data_std < 1e-4)[0])
	dimensions_to_use = list(np.where(data_std >= 1e-4)[0])[6:]

	print 'Used dimensions:', len(dimensions_to_use), dimensions_to_use
	# For euler:
	# 48, [6,7,8,9,12,13,14,15,21,22,23,24,27,28,29,30,36,37,38,39,40,41,42,43,44,45,46,47,51,52,53,54,55,56,57,60,61,62,75,76,77,78,79,80,81,84,85,86]

	data_std[dimensions_to_ignore] = 1.0

	with open(DATA_DIR+to_type+'/stats.json', 'wb') as param_file:
	 	json.dump({
	 		'data_mean':data_mean.tolist(),
	 		'data_std':data_std.tolist(),
	 		'dim_to_ignore':dimensions_to_ignore,
	 		'dim_to_use':dimensions_to_use,
	 		 # this is added for convenience
			'action_list':{a:i for i,a in enumerate(ACTIONS)}
	 	}, param_file)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	list_of_type = ['euler', 'euclidean', 'expmap']
	ap.add_argument('-t', '--type', required=False, help='Choice of parameterization', default='euler', choices=list_of_type)
	ap.add_argument('-v', '--visualize', action='store_true', help='Visualize the data only')

	args = vars(ap.parse_args())

	#convert(args['type'], args['visualize'])
	if not args['visualize']:
		whitening(args['type'])
