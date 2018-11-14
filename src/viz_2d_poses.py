import numpy as np
import matplotlib.pyplot as plt

from utils import converter, utils

# same as https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py#L21
M_POSE_LINES = {'r':[0,1,2,3],
		'g':[0,6,7,8],
		'b':[0,12,13,14,15],
		'm':[13,17,18,19],
		'k':[13,25,26,27]}

def __add_line(plt_canvas, coords, color, size, s=0):
	coords[:,1] = coords[:,1] + 350*s
	plt_canvas.plot(coords[:,1], coords[:,2], color=color, linewidth=size)


def __plot_overlaying_batch(motions, titles, param_type):
        '''
        Plot poses.
        '''
        n,t,_ = motions.shape
	wspace = np.max(np.abs(motions))/2
        f, axarr = plt.subplots(n, 1, sharex=True, sharey=True)
        #f.patch.set_visible(False)
        f.subplots_adjust(wspace=-0.1)
        for i in range(n):
                # add title
		#axarr[i].axis('off')
		axarr[i].spines["top"].set_visible(False)
		axarr[i].spines["right"].set_visible(False)
		axarr[i].spines["bottom"].set_visible(False)
		axarr[i].spines['left'].set_visible(False)
		axarr[i].get_yaxis().set_ticks([])
		axarr[i].get_xaxis().set_ticks([])
		print titles[i].title()

                x = motions[i]
                # convert to euclidean if needed
                if param_type == 'euler':
                        x = converter.sequence_euler2xyz(x)
                elif param_type == 'expmap':
                        x = converter.sequence_expmap2xyz(x)
                x = np.reshape(x, (t,-1,3))
                # plot each frame
                for j in range(t):
                        for c, l in M_POSE_LINES.iteritems():
                                __add_line(axarr[i], x[j][l], c, 1, j)

        f.subplots_adjust(hspace=0.1)
        return f, axarr


def __plot_batch(motions, titles, param_type):
	'''
	Plot poses.
	'''
	n,t,_ = motions.shape
	f, axarr = plt.subplots(n, t, sharex=True, sharey=True)
	#f.patch.set_visible(False)
	f.subplots_adjust(wspace=-0.1)
	for i in range(n):
		# add title
		if len(titles) > 0:
			axarr[i, 0].set_ylabel(titles[i], rotation=0, labelpad=20)

		x = motions[i]
		# convert to euclidean if needed
		if param_type == 'euler':
			x = converter.sequence_euler2xyz(x)
		elif param_type == 'expmap':
			x = converter.sequence_expmap2xyz(x)
		x = np.reshape(x, (t,-1,3))
		# plot each frame
		for j in range(t):
			axarr[i,j].axis('off')
			# remove ticks
			axarr[i, j].get_xaxis().set_ticks([])
			axarr[i, j].get_yaxis().set_ticks([])
			for c, l in M_POSE_LINES.iteritems():
				__add_line(axarr[i, j], x[j][l], c, 1)

	f.subplots_adjust(hspace=0.1)
	return f, axarr

def plot_batch(motions, stats, args, **kwargs):
	param_type = kwargs['parameterization'] if 'parameterization' in kwargs else 'euclidea'
	titles = kwargs['row_titles'] if 'row_titles' in kwargs else []
	title = kwargs['title'] if 'title' in kwargs else 'plot'

	if param_type != 'euclidean':
		motions = utils.recover(motions, stats)

	f, axarr = __plot_overlaying_batch(motions, titles, param_type)
		#__plot_batch(motions, titles, param_type)

	if title is not None:
		plt.suptitle(title.title())

	if args['debug']:
		plt.show()
	else:
		filename = args['output_dir'] + '_' + title.lower().replace(' ', '_') + '.png'
		f.savefig(filename)
		plt.close(f)
		print 'Saved to', filename
