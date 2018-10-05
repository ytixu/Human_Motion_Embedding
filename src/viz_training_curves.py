'''
For plotting training curves (code in src/utils/viz_training_curves.py),
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

from utils import viz_training_curves

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', required=True,
		help='Path to the .csv file')
	ap.add_argument('-i', '--iterations', required=False,
		help='Plot only this many iterations', default=-1, type=int)
	ap.add_argument('-s', '--save', required=False,
		help='Save plot to the same directory as the input file',
		action='store_true')
	args = vars(ap.parse_args())

	viz_training_curves.plot_training_curve(args)

# python viz_training_curves.py -p ../out/gru_t40_l1024_u10_loss-mean_absolute_error_opt-Nadam-lr=0.001_norm_pi_1538602020.csv -i 150