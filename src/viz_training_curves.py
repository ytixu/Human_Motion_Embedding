'''
For plotting training curves,
input log file that was outputted from src/train.py, where all errors
are computed using l2 of the euler angle.

Plot lines:
	- training error
	- test error
	- short term motion prediction error (80ms, 160ms, 360ms and 400ms)

If we are using other methods than Seq2Seq and H_Seq2Seq, we also have
	- the mean of the standard deviation of the modality difference
	- the standard deviation of the standard deviation of the modality difference

Also print the prediction performance at the best training error and test error
you can compare it with the best prediction performance seen during training
'''

import argparse
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt

from utils.utils import SHORT_TERM_IDX

AVERAGE_OVER = 3

LABELS = [['Train','Test'],
			['80ms', '160ms', '320ms', '400ms'],
			['Mean STD (x10)','STD STD (x10)']]
FILLED_MARKER = ['o', 'v', '^', '*']
LINESTYLES = ['-', '--', ':']
COLOR_STYLE = ['#1f77b4','#ff7f0e','#2ca02c']

ACTIONS = {"purchases": 8, "walking": 0, "takingphoto": 11, "eating": 1, "sitting": 9, "discussion": 3, "walkingdog": 13, "greeting": 5, "walkingtogether": 14, "phoning": 6, "posing": 7, "directions": 4, "smoking": 2, "waiting": 12, "sittingdown": 10}

def __to_numb_array(row):
	a = row.replace('\n', '').replace(',', ' ').strip('[').strip(']').split()
	a = [i for i in a if i != '']
	return np.array(map(float, a))

def __classification_score_per_action(data):
	N = 8
	return [np.mean(data[i*N:(i+1)*N]) for action, i in ACTIONS.iteritems()]

def __get_data(row, for_classification):
	train = np.mean(__to_numb_array(row[1]))
	test = np.mean(__to_numb_array(row[2]))
	valid, std_mean, std_std = 0,0,0
	if for_classification:
		valid = __classification_score_per_action(__to_numb_array(row[6]))
		std_mean = float(row[3])
		std_std = float(row[4])
	else:
		valid = __to_numb_array(row[3])[SHORT_TERM_IDX]
		std_mean = float(row[4])*10
		std_std = float(row[5])*10
	return [[train, test], valid, [std_mean, std_std]]

def __print_format_performance(line, for_classification):
	print 'Training: %f'%(line[0][0])
	print 'Testing: %f'%(line[0][1])
	if for_classification:
		for i,a in enumerate(ACTIONS):
			print a, line[1][i]
	else:
		print 'Prediction (80-160-320-400ms): '+' '.join(map(str,line[1]))
	print 'MEAN:', np.mean(line[1])
	print 'MEAN STD: %f\nSTD STD: %f' %(line[2][0], line[2][1])

def plot_training_curve(args):
	global LABELS

	data = None
	filename = args['path']
	name = '.'.join(os.path.basename(filename).split('.')[:-1])
	directory = os.path.dirname(filename)

	iter_n = args['iterations']
	plot_classification = args['classification']

	# Reading data
	with open(filename) as csvfile:
		spamreader = csv.reader(csvfile)
		data = [__get_data(row, plot_classification) for row in spamreader]
		if iter_n > -1:
			data = data[:iter_n]

	key = len(data[0])

	# Get best performance
	best_line_idx = np.argmin([np.mean(d[1]) for d in data])
	print 'BEST VALIDATION SCORE:'
	__print_format_performance(data[best_line_idx], plot_classification)
	best_line_idx = np.argmin([np.mean(d[0]) for d in data])
	print 'BEST PERFORMANCE:'
	__print_format_performance(data[best_line_idx], plot_classification)

	# Plot curves
	if plot_classification:
		actions = {v:k for k,v in ACTIONS.iteritems()}
		avg_valid = [np.mean([data[i][1][j] for i in range(len(data))]) for j in range(len(data[0][1]))]
		top_3_classes = np.argsort(avg_valid)[:3]
		LABELS[1] = [actions[i] for i in top_3_classes] + ['mean_class']
		for i in range(len(data)):
			mean_class = np.mean(data[i][1])
			data[i][1] = [data[i][1][j] for j in top_3_classes] + [mean_class]

	fig = plt.figure()
	ax = plt.subplot(111)

	for i in range(key): # for train-test, valid, std_mean-std_std
		if i == 2:
			continue
		for j in range(min(len(data[0][i]), 4)):
			line = [np.mean([data[kk][i][j]
						for kk in range(max(0,k-AVERAGE_OVER), min(len(data),k+AVERAGE_OVER))])
					for k in range(0,len(data),10)]
			if np.mean(line) == 0: # skip std for Seq2Seq and H_Seq2Seq
				continue
			x = range(0,len(line)*10,10)
			ax.plot(x, line, label=LABELS[i][j],
				marker=FILLED_MARKER[j], linestyle=LINESTYLES[i],
				c=COLOR_STYLE[i], linewidth=3, markersize=10)

	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	# Put a legend below current axis
	ax.legend()#loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Mean angle error')

	# plt.suptitle(name)

	# Show or save plot
	if args['save']:
		filename = directory+'/'+name+'.png'
		fig.savefig(filename)
		plt.close(fig)
		print 'Saved to',filename
	else:
		plt.show()

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', required=True,
		help='Path to the .csv file', nargs = '*')
	ap.add_argument('-i', '--iterations', required=False,
		help='Plot only this many iterations', default=-1, type=int)
	ap.add_argument('-s', '--save', required=False,
		help='Save plot to the same directory as the input file',
		action='store_true')
	ap.add_argument('-c', '--classification', required=False,
		help='Plot classification error instead of prediction error.',
		action='store_true')
	args = vars(ap.parse_args())

	files = args['path']
	for f in files:
		args['path'] = f
		plot_training_curve(args)

# python viz_training_curves.py -p ../out/gru_t20_l512_u10_loss-mean_squared_error_opt-optimizers.Nadam-lr=0.001_norm_pi_1539338293.csv -i 150
