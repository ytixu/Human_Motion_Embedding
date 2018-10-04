import argparse
import csv
import os.path

import numpy as np
import matplotlib.pyplot as plt

from utils import SHORT_TERM_IDX


def __to_numb_array(row):
	a = row.replace('\n', '').strip('[').strip(']').split()
	a = [i for i in a if i != '']
	return np.array(map(float, a))

def __get_data(row):
	if len(row) == 3:
		train = __to_numb_array(row[1])[SHORT_TERM_IDX]
		valid = __to_numb_array(row[2])[SHORT_TERM_IDX]
		return train, valid

def __plot_training_curve(args):
	data = None
	filename = args['path']
	name = os.path.basename(filename).split('.')[0]
	directory = os.path.dirname(filename)

	iter_n = args['iterations']

	with open(filename) as csvfile:
		spamreader = csv.reader(csvfile)
		data = [__get_data(row) for row in spamreader]
		if iter_n > -1:
			data = data[:iter_n]

	fig = plt.figure()
	ax = plt.subplot(111)
	labels = {2:(['train-80', 'train-160', 'train-320', 'train-400'],
				['valid-80', 'valid-160', 'valid-320', 'valid-400']),
			3:(['train'],['valid-80', 'valid-160', 'valid-320', 'valid-400'],
				['test'])}

	colors = ['b','r','g','k']
	linestyles = ['-','--','-']

	key = len(data[0])
	for i in range(key): # for train, valid
		for j in range(len(data[0][i])): # for each time frames
			line = [data[k][i][j] for k in range(len(data))]
			x = range(len(line))
			ax.plot(x, line, label=labels[key][i][j],
				c=colors[j], linestyle=linestyles[i])

	# Shrink current axis's height by 10% on the bottom
	box = ax.get_position()
	ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
	# Put a legend below current axis
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
	plt.suptitle(name)
	# fig.savefig(directory+'/'+name+'.png')
	# plt.close(fig)
	plt.show()



if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', required=True, help='Path to the .csv file')
	ap.add_argument('-i', '--iterations', required=False, help='Plot only this many iterations', default=-1, type=int)
	args = vars(ap.parse_args())

	__plot_training_curve(args)

	# python viz_training_curves.py -p ../../out/gru_t40_l1024_u10_loss-mean_absolute_error_opt-Nadam-lr=0.001_norm_pi_1538602020.csv -i 150