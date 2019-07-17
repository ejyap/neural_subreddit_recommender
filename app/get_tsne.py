from MulticoreTSNE import MulticoreTSNE as TSNE
import h5py

hf = h5py.File('../data/neumf_weights.h5')
weights = hf.get('neumf_weights')[()]

weights_tsne = TSNE(n_components=3, metric ='cosine', n_jobs=4).fit_transform(weights)

h5f = h5py.File('../data/neumf_tsne_weights.h5', 'w')
h5f.create_dataset('tsne_weights', data=weights_tsne)
h5f.close()