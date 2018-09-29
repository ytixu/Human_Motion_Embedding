import argparse
import os
import numpy as np

import converter

DATA_DIR = '../../data/h3.6m/dataset'

def create_dir(directory):
	if not os.path.exists(directory):
	    os.makedirs(directory)

TRAIN_SUBJECT_ID = [1,6,7,8,9,11]
TEST_SUBJECT_ID = [5]
ACTIONS = ["walking", "eating", "smoking", "discussion",  "directions",
              "greeting", "phoning", "posing", "purchases", "sitting",
              "sittingdown", "takingphoto", "waiting", "walkingdog",
              "walkingtogether"]

def readCSVasFloat(filename):
	"""
	Borrowed from SRNN code. Reads a csv and returns a float matrix.
	https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
	-------
	Args
		filename: string. Path to the csv file
	Returns
		returnArray: the read data in a float32 matrix
	"""
	returnArray = []
	lines = open(filename).readlines()
	for line in lines:
		line = line.strip().split(',')
		if len(line) > 0:
			returnArray.append(np.array([np.float32(x) for x in line]))

	returnArray = np.array(returnArray)
	return returnArray

def load_data(path_to_dataset, subjects, actions):
	"""
	una-dinosauria/human-motion-prediction: Borrowed from SRNN code.
		This is how the SRNN code reads the provided .txt files
		https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270
	ytixu/Human_Motion_Embedding: Borrowed again and modified from:
		https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L216
	-------
	Args
		path_to_dataset: string. directory where the data resides
		subjects: list of numbers. The subjects to load
		actions: list of string. The actions to load
	Returns
		trainData: dictionary with k:v
		k=(subject, action, subaction, 'even'), v=(nxd) un-normalized data
		completeData: nxd matrix with all the data. Used to normlization stats
 	"""
	for subj in subjects:
		for action_idx in np.arange(len(actions)):

			action = actions[ action_idx ]

			for subact in [1, 2]:  # subactions

				print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

				filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
				action_sequence = readCSVasFloat(filename)

				n, d = action_sequence.shape
				even_list = range(0, n, 2)

				yield (subj, action, subact), action_sequence[even_list, :]

def convert(to_type='euler'):
	train_set = load_data(DATA_DIR, TRAIN_SUBJECT_ID, ACTIONS)
  	test_set = load_data(DATA_DIR, TEST_SUBJECT_ID,  ACTIONS)

  	directory = to_type
  	create_dir(directory+'/train/')
  	create_dir(directory+'/test/')

	for data_name, data_set in ({'train':train_set, 'test':test_set}).iteritems():
		for k, x in data_set:
			print 'save', k, x.shape
			subject, action, subact = k
			data = None

			if to_type == 'expmap':
				data = x
			elif to_type == 'euler':
				data = converter.sequence_expmap2euler(x)
				# converter.animate(converter.sequence_expmap2euler(data)) # Uncomment this to visualize the data
			else:
				data = converter.sequence_expmap2xyz(x)
				# converter.animate(data) # Uncomment this to visualize the data

			print data.shape
			# np.save(directory+'/%s/%s_%d_%d.npy'%(data_name, action, subject, subact), data)


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	list_of_type = ['euler', 'euclidean', 'expmap']
	ap.add_argument('-t', '--type', required=False,
		help='Choice of parameterization', default='euler', choices=list_of_type)
	args = vars(ap.parse_args())

	convert(args['type'])