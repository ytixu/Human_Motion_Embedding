import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from utils import converter, utils

M_POSE_LINES = {'r':[0,1,2,3],
		'g':[0,4,5,6],
		'b':[0,7,8,9,10],
		'm':[8,11,12,13],
		'k':[8,14,15,16]}

class Ax3DPose(object):
	def __init__(self, ax):
		self.ax = ax
		vals = np.zeros((17, 3))

		# Make connection matrix
		self.plots = []
		for i, lines in enumerate(M_POSE_LINES.iteritems()):
			color, line = lines
			x = np.array(vals[line, 0])
			y = np.array(vals[line, 1])
			z = np.array(vals[line, 2])
			self.plots.append(self.ax.plot(x, y, z, lw=2, c=color))

		# self.ax.set_xlabel("x")
		# self.ax.set_ylabel("y")
		# self.ax.set_zlabel("z")
		self.ax.set_xlim3d([-1000, 1000])
		self.ax.set_zlim3d([-1000, 1000])
		self.ax.set_ylim3d([-1000, 1000])
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_zticks([])

	def update(self, channels, forced_color=None):
		"""
		Update the plotted 3d pose.
		Args
			channels: 96-dim long np array. The pose to plot.
			lcolor: String. Colour for the left part of the body.
			rcolor: String. Colour for the right part of the body.
		Returns
			Nothing. Simply updates the axis with the new pose.
		"""

		vals = np.reshape( channels, (-3, 3) )[[0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]]

		for i, lines in enumerate(M_POSE_LINES.iteritems()):
			color, line = lines
			x = np.array(vals[line, 0])
			y = np.array(vals[line, 1])
			z = np.array(vals[line, 2])
			self.plots[i][0].set_xdata(x)
			self.plots[i][0].set_ydata(y)
			self.plots[i][0].set_3d_properties(z)
			if forced_color is not None:
				self.plots[i][0].set_color(forced_color)
			else:
				self.plots[i][0].set_color(color)

		self.ax.set_aspect('equal')

	def get_lines(self):
		return [self.plots[i][0] for i in range(len(M_POSE_LINES))]

def animate_motion(seq, name, save_path, **kwargs):
	# convert to euclidean if needed
	if 'param_type' in kwargs and kwargs['param_type'] != 'euclidean':
		seq = utils.recover(np.array([seq]), kwargs['stats'])[0]
		if kwargs['param_type'] == 'euler':
			seq = converter.sequence_euler2xyz(seq)
		elif kwargs['param_type'] == 'expmap':
			seq = converter.sequence_expmap2xyz(seq)

	if type(name) == type('str'):
		seq = [seq]
		name = [name]

	fig = plt.figure()
	n = len(seq)
	axs = [None]*n
	obs = [None]*n
	for i in range(n):
		axs[i] = fig.add_subplot(1, len(seq), i+1, projection='3d')
		#axs[i].set_title(name[i])
		obs[i] = Ax3DPose(axs[i])

	n_t = seq[0].shape[0]

	def init():
		if n == 1:
			return obs[0].get_lines()
		return reduce(lambda acc, x: acc + x, [obs[i].get_lines() for i in range(n)])

	def animate(t):
		for i in range(n):
			obs[i].update(seq[i][t])
		return init()


	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_t, interval=400, blit=True)
	filename = save_path+'-'.join(name)+'.gif'
	anim.save(filename, writer='imagemagick', fps=60)


def animate_compare(action, stats, save_path, param_type, **kwargs):

	# start_seq, true_seq, pred_seq, pred_name, baseline_seq, baseline_name, 

	fig = plt.figure()
	ax_true = fig.add_subplot(1, 3, 1, projection='3d')
	ax_baseline = fig.add_subplot(1, 3, 2, projection='3d')
	ax_pred = fig.add_subplot(1, 3, 3, projection='3d')

	ax_true.set_title('Our-FN')#Ground Truth')
	ax_baseline.set_title(kwargs['baseline_name'])
	ax_pred.set_title(kwargs['pred_name'])

	ob_true = Ax3DPose(ax_true)
	ob_baseline = Ax3DPose(ax_baseline)
	ob_pred = Ax3DPose(ax_pred)

	n_start = kwargs['start_seq'].shape[0]

	# convert to euclidean if needed
	if param_type != 'euclidean':
		for k in kwargs:
			if type(kwargs[k]) == type('str'):
				continue
			if kwargs[k].shape[0] == 0:
				continue
			kwargs[k] = utils.recover(np.array([kwargs[k]]), stats)[0]
			if param_type == 'euler':
				kwargs[k] = converter.sequence_euler2xyz(kwargs[k])
			elif param_type == 'expmap':
				kwargs[k] = converter.sequence_expmap2xyz(kwargs[k])

	def init():
		return ob_true.get_lines() + ob_pred.get_lines() + ob_pred.get_lines()

	N = 1
	def animate(t):
		t = t/N
		if t >= n_start:
			ob_true.update(kwargs['true_seq'][t-n_start])
			ob_baseline.update(kwargs['baseline_seq'][t-n_start])
			ob_pred.update(kwargs['pred_seq'][t-n_start])
		else:
			ob_true.update(kwargs['start_seq'][t], '#000000')
			ob_baseline.update(kwargs['start_seq'][t], '#000000')
			ob_pred.update(kwargs['start_seq'][t], '#000000')
		return init()


	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=(kwargs['true_seq'].shape[0]+n_start)*N, interval=400, blit=True)
	filename = save_path + action + '_anim.gif'
	anim.save(filename, writer='imagemagick', fps=15)
