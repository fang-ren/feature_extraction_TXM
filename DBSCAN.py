"""
author: Fang Ren (SSRL)

5/2/2017
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import os.path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import scale


raw_path = 'data\\Imgstack.mat'
nor_path = 'data\\Imgstack_Nor.mat'
energy_path = 'data\\Imgstack_energy.mat'
feature_path = 'data\\features\\'

# import spectra
images = loadmat(raw_path)['Imgstack']
images_norm = loadmat(nor_path)['Nor']
energies = loadmat(energy_path)['Es']

# import features
edgejumpmap = loadmat(feature_path+ 'edgejumpmap.mat')['edgejumpmap']
edgeposition_stack = loadmat(feature_path+ 'edgeposition_stack.mat')['edgeposition_stack']
peak_stack = loadmat(feature_path+ 'peak_stack.mat')['peak_stack']
Goodness_of_fit = loadmat(feature_path+ 'Goodness_of_fit.mat')['R_squares_stack']
noisemap_stack = loadmat(feature_path+ 'noisemap_stack.mat')['noisemap_stack']
#features = np.concatenate(([edgejumpmap], [edgeposition_stack], [peak_stack], [Goodness_of_fit], [noisemap_stack]), axis = 0)


# compress the image for testing
keep = np.arange(0, 400, 10)
edgejumpmap = edgejumpmap[keep][:,keep]
edgeposition_stack = edgeposition_stack[keep][:,keep]
peak_stack = peak_stack[keep][:,keep]
Goodness_of_fit = Goodness_of_fit[keep][:,keep]
noisemap_stack = noisemap_stack[keep][:,keep]
images_norm = images_norm[keep][:,keep]


# get the dimension of the image
s1 = edgejumpmap.shape[0]
s2 = edgejumpmap.shape[1]

# flatten image
edgejumpmap = edgejumpmap.reshape(s1*s2, 1)
edgeposition_stack = edgeposition_stack.reshape(s1*s2, 1)
peak_stack = peak_stack.reshape(s1*s2, 1)
Goodness_of_fit = Goodness_of_fit.reshape(s1*s2, 1)
noisemap_stack = noisemap_stack.reshape(s1*s2, 1)
features = np.concatenate((edgejumpmap, edgeposition_stack, peak_stack, Goodness_of_fit, noisemap_stack), axis = 1)

# # standardize feature scale: (value - mean)/sigma
# features = scale(features, axis = 0)

# DBSCAN
db = DBSCAN(eps = 0.5, min_samples= 5)
labels = db.fit_predict(features)

# # agglomerative clustering
# ac = AgglomerativeClustering(n_clusters = 5)
# labels = ac.fit_predict(features)
#
# # spectral clustering
# sc = SpectralClustering(n_clusters = 5)
# labels = sc.fit_predict(features)


# reshape the 1D array into images
labels = labels.reshape(s1, s2)
edgejumpmap = features[:,0].reshape(s1 ,s2)
edgeposition_stack = features[:,1].reshape(s1, s2)
peak_stack = features[:,2].reshape(s1, s2)
Goodness_of_fit = features[:,3].reshape(s1, s2)
noisemap_stack = features[:,4].reshape(s1, s2)

# visualization
save_path = 'report\\'
plt.figure(1, figsize=(12,12))
plt.subplot(331)
plt.title('edge jump')
plt.imshow(edgejumpmap)
plt.colorbar()
plt.subplot(332)
plt.title('edge position')
plt.imshow(edgeposition_stack)
plt.clim(7650, 7982)
plt.colorbar()
plt.subplot(333)
plt.title('first peak position')
plt.imshow(peak_stack)
plt.clim(7650, 7742)
plt.colorbar()
plt.subplot(334)
plt.title('goodness of fit')
plt.imshow(Goodness_of_fit)
plt.colorbar()
plt.subplot(335)
plt.title('pre-edge noise')
plt.imshow(noisemap_stack)
plt.colorbar()
plt.subplot(336)
plt.title('clustering')
plt.imshow(labels)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering.png'))