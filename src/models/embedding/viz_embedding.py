import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from scipy.spatial import ConvexHull

def __pca_reduce(embedding):
	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	X_norm = (embedding - embedding.min())/(embedding.max() - embedding.min())
	transformed = pca.fit_transform(X_norm)
	return transformed

def plot_convex_hall(embedding, args):
	# sort all the modalities and concat embedding into one list
	modalities = sorted(embedding.keys())
	sample_n = len(embedding[modalities[0]])
	sorted_emb = np.concatenate([embedding[m] for m in modalities], axis=0)

	transformed = __pca_reduce(sorted_emb)

	# plot convex hall
	for i, m in enumerate(modalities):
		choices = transformed[i*sample_n:(i+1)*sample_n]
		hull = ConvexHull(choices)
		for simplex in hull.simplices:
			plt.plot(choices[simplex, 0], choices[simplex, 1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)
		# plt.scatter(choices[:,0], choices[:,1], c='#'+('%s'%('{0:01X}'.format(i+2)))*6)

	# for c in ['red', 'green', 'blue']:
	# 	n = np.random.randint(0, sample_n-50)
	# 	for i, s in enumerate(['x', '+', 'd']):
	# 		random_sequence = (transformed[i*(h-1)/2::h])[n:n+50:10] if model_name == H_LSTM else (transformed[i*(h-1)/2*sample_n:])[n:n+50:10]
	# 		plt.scatter(random_sequence[:,0], random_sequence[:,1], c=c, marker=s)

	plt.legend(loc=3)
	if args['debug']:
		plt.show()
	else:
		plt.savefig(args['output_dir']+'plot_convex_hall.png')
		plt.close()