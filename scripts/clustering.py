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
from scipy.signal import medfilt

raw_path = '..\\data\\Imgstack.mat'
nor_path = '..\\data\\Imgstack_Nor.mat'
energy_path = '..\\data\\Imgstack_energy.mat'
feature_path = '..\\data\\features_2\\'

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

# # apply median filter to remove noise
# kernel = 7
# edgejumpmap = medfilt(edgejumpmap, kernel_size= kernel)
# edgeposition_stack = medfilt(edgeposition_stack, kernel_size= kernel)
# peak_stack = medfilt(peak_stack, kernel_size= kernel)
# Goodness_of_fit = medfilt(Goodness_of_fit, kernel_size= kernel)
# noisemap_stack = medfilt(noisemap_stack, kernel_size= kernel)


# compress the image for testing
keep = np.arange(0, 400, 10)
edgejumpmap = edgejumpmap[keep][:,keep]
edgeposition_stack = edgeposition_stack[keep][:,keep]
peak_stack = peak_stack[keep][:,keep]
Goodness_of_fit = Goodness_of_fit[keep][:,keep]
noisemap_stack = noisemap_stack[keep][:,keep]
images_norm = images_norm[keep][:,keep]

# plt.imshow(Goodness_of_fit)
# plt.savefig('Goodness_of_fit.png')
# plt.close('all')

# applying mask, when the value is zero
edgejumpmap[(edgejumpmap == 0)] = np.nan
edgeposition_stack[edgeposition_stack == 0 * (abs(edgeposition_stack - np.mean(edgeposition_stack)) > 3 * np.std(edgeposition_stack))] = np.nan
peak_stack[peak_stack == 0] = np.nan
Goodness_of_fit[Goodness_of_fit == 0] = np.nan
noisemap_stack[noisemap_stack == 0] = np.nan

# remove outliers using 3 sigma
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
peak_stack[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan

# plt.imshow(Goodness_of_fit)
# plt.savefig('Goodness_of_fit3.png')
# plt.close('all')


# print edgejumpmap

# get the dimension of the image
s1 = edgejumpmap.shape[0]
s2 = edgejumpmap.shape[1]

# flatten image
edgejumpmap = edgejumpmap.reshape(s1*s2, 1)
edgeposition_stack = edgeposition_stack.reshape(s1*s2, 1)
peak_stack = peak_stack.reshape(s1*s2, 1)
Goodness_of_fit = Goodness_of_fit.reshape(s1*s2, 1)
noisemap_stack = noisemap_stack.reshape(s1*s2, 1)

# standardize the data
edgejumpmap = (edgejumpmap - np.nanmean(edgejumpmap))/np.nanstd(edgejumpmap)
edgeposition_stack = (edgeposition_stack - np.nanmean(edgeposition_stack))/np.nanstd(edgeposition_stack)
peak_stack = (peak_stack - np.nanmean(peak_stack))/np.nanstd(peak_stack)
Goodness_of_fit = (Goodness_of_fit - np.nanmean(Goodness_of_fit))/np.nanstd(Goodness_of_fit)
noisemap_stack = (noisemap_stack - np.nanmean(noisemap_stack))/np.nanstd(noisemap_stack)


# incoproate all features in to a box
features = np.concatenate((edgejumpmap, edgeposition_stack, peak_stack, Goodness_of_fit, noisemap_stack), axis = 1)

# replace nan with zeros
features = np.nan_to_num(features)

# # standardize feature scale: (value - mean)/sigma
# features = scale(features, axis = 0)

# # DBSCAN
# db = DBSCAN(eps = 0.8, min_samples= 3)
# labels = db.fit_predict(features)

# # agglomerative clustering
# ac = AgglomerativeClustering(n_clusters = 4)
# labels = ac.fit_predict(features)

# spectral clustering
sc = SpectralClustering(n_clusters = 4)
labels = sc.fit_predict(features)


# reshape the 1D array into images
labels = labels.reshape(s1, s2)
edgejumpmap = features[:,0].reshape(s1 ,s2)
edgeposition_stack = features[:,1].reshape(s1, s2)
peak_stack = features[:,2].reshape(s1, s2)
Goodness_of_fit = features[:,3].reshape(s1, s2)
noisemap_stack = features[:,4].reshape(s1, s2)

# visualization
save_path = '..\\report\\'
plt.figure(1, figsize=(12,8))
plt.subplot(231)
plt.title('edge jump')
plt.imshow(edgejumpmap)
plt.colorbar()
plt.subplot(232)
plt.title('edge position')
plt.imshow(edgeposition_stack)
#plt.clim(7650, 7982)
plt.colorbar()
plt.subplot(233)
plt.title('first peak position')
plt.imshow(peak_stack)
#plt.clim(7650, 7742)
plt.colorbar()
plt.subplot(234)
plt.title('goodness of fit')
plt.imshow(Goodness_of_fit)
plt.colorbar()
plt.subplot(235)
plt.title('pre-edge noise')
plt.imshow(noisemap_stack)
plt.colorbar()
plt.subplot(236)
plt.title('clustering')
plt.imshow(labels)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering.png'))