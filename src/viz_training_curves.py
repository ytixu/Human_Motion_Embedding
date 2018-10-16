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

LABELS = [['train','test'],
			['valid-80', 'valid-160', 'valid-320', 'valid-400'],
			['mean-std','std-std']]
COLORS = ['b','r','g','k']
LINESTYLES = ['-', '--', ':']

def __to_numb_array(row):
	a = row.replace('\n', '').replace(',', ' ').strip('[').strip(']').split()
	a = [i for i in a if i != '']
	return np.array(map(float, a))

def __get_data(row):
	train = np.mean(__to_numb_array(row[2]))
	test = np.mean(__to_numb_array(row[3]))
	valid = __to_numb_array(row[4])[[i+int(row[0]) for i in SHORT_TERM_IDX]] # TODO: fix this
	std_mean = float(row[5])
	std_std = float(row[6])
	return [train, test], valid, [std_mean, std_std]

def __format_performance(line):
	return '\nTraining: %f\nTesting: %f\nPrediction (80-160-320-400ms): '%(
		line[0][0], line[0][1])+' '.join(map(str,
			line[1]))+'\nMEAN STD: %f\nSTD STD: %f' %(line[2][0], line[2][1])

def plot_training_curve(args):
	data = None
	filename = args['path']
	name = '.'.join(os.path.basename(filename).split('.')[:-1])
	directory = os.path.dirname(filename)

	iter_n = args['iterations']

	# Reading data
	with open(filename) as csvfile:
		spamreader = csv.reader(csvfile)
		data = [__get_data(row) for row in spamreader]
		if iter_n > -1:
			data = data[:iter_n]

	key = len(data[0])

	# Get best performance
	best_line_idx = np.argmin([np.mean(d[1]) for d in data])
	print 'BEST VALIDATION SCORE:', __format_performance(data[best_line_idx])
	best_line_idx = np.argmin([np.mean(d[0]) for d in data])
	print 'BEST PERFORMANCE:', __format_performance(data[best_line_idx])

	# Plot curves
	fig = plt.figure()
	ax = plt.subplot(111)

	for i in range(key): # for train-test, valid, std_mean-std_std
		for j in range(len(data[0][i])): # for each time frames
			line = [data[k][i][j] for k in range(len(data))]
			if np.mean(line) == 0: # skip std for Seq2Seq and H_Seq2Seq
				continue
			x = range(len(line))
			ax.plot(x, line, label=LABELS[i][j],
				c=COLORS[j], linestyle=LINESTYLES[i])

	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	# Put a legend below current axis
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
	plt.suptitle(name)

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
	args = vars(ap.parse_args())

	files = args['path']
	for f in files:
		args['path'] = f
		plot_training_curve(args)

# python viz_training_curves.py -p ../out/gru_t40_l1024_u10_loss-mean_absolute_error_opt-Nadam-lr=0.001_norm_pi_1538602020.csv -i 150
