"""
author: Fang Ren (SSRL)

5/2/2017
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from import_features import import_features
from compress_image import compress_image
from zero_mask import zero_mask
from outlier_mask import outlier_mask
from flatten_imArray import flatten_imArray

#########################
# user input
edgejumpmap_mode = 'off'
edgeposition_stack_mode = 'on'
goodness_of_fit_mode = 'on'
peak_height_mode = 'on'
peak_stack_mode = 'on'
noisemap_stack_mode = 'on'
#########################

#########################
# import 1st particle
feature_path1 = '..\\data\\particle1\\'
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = import_features(feature_path1)
#########################

#########################
# import 2nd particle
feature_path2 = '..\\data\\particle2\\'
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = import_features(feature_path2)
#########################

#########################
# compress images for speeding up, uncomment for real results
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = compress_image(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = compress_image(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################

#########################
# applying zero mask for both particles
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = zero_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = zero_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################

#########################
# applying outlier mask for both particles
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = outlier_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = outlier_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################


#########################
# visualizing features
save_path = '..\\report\\feature_visualization\\'
# particle 1
plt.figure(1)
plt.title('edge jump')
plt.imshow(np.nan_to_num(edgejumpmap1), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge jump_particle1.png'))
plt.figure(2)
plt.title('edge position')
plt.imshow(np.nan_to_num(edgeposition_stack1), cmap = 'viridis')
plt.clim(7650, 7982)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge position_particle1.png'))
plt.figure(3)
plt.title('first peak position')
plt.imshow(np.nan_to_num(peak_stack1), cmap = 'viridis')
plt.clim(7650, 7742)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'first peak position_particle1.png'))
plt.figure(4)
plt.title('goodness of fit')
plt.imshow(np.nan_to_num(goodness_of_fit1), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'goodness of fit_particle1.png'))
plt.figure(5)
plt.title('pre-edge noise')
plt.imshow(np.nan_to_num(noisemap_stack1), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'pre-edge noise_particle1.png'))
plt.figure(6)
plt.title('peak height')
plt.imshow(np.nan_to_num(peak_height1), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'peak height_particle1.png'))

# particle 2
plt.figure(7)
plt.title('edge jump')
plt.imshow(np.nan_to_num(edgejumpmap2), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge jump_particle2.png'))
plt.figure(8)
plt.title('edge position')
plt.imshow(np.nan_to_num(edgeposition_stack2), cmap = 'viridis')
plt.clim(7650, 7982)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge position_particle2.png'))
plt.figure(9)
plt.title('first peak position')
plt.imshow(np.nan_to_num(peak_stack2), cmap = 'viridis')
plt.clim(7650, 7742)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'first peak position_particle2.png'))
plt.figure(10)
plt.title('goodness of fit')
plt.imshow(np.nan_to_num(goodness_of_fit2), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'goodness of fit_particle2.png'))
plt.figure(11)
plt.title('pre-edge noise')
plt.imshow(np.nan_to_num(noisemap_stack2), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'pre-edge noise_particle2.png'))
plt.figure(12)
plt.title('peak height')
plt.imshow(np.nan_to_num(peak_height2), cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'peak height_particle2.png'))

plt.close('all')
#########################

#########################
# flatten images
s1 = edgejumpmap1.shape[0]
s2 = edgejumpmap1.shape[1]
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################

#########################
# conbine two particle data
edgejumpmap = np.concatenate((edgejumpmap1,edgejumpmap2))
edgeposition_stack = np.concatenate((edgeposition_stack1, edgeposition_stack2))
peak_stack = np.concatenate((peak_stack1, peak_stack2))
goodness_of_fit = np.concatenate((goodness_of_fit1, goodness_of_fit2))
noisemap_stack = np.concatenate((noisemap_stack1, noisemap_stack2))
peak_height = np.concatenate((peak_height1, peak_height2))
##########################

##########################
# standardized features
edgejumpmap = (edgejumpmap - np.nanmean(edgejumpmap))/np.nanstd(edgejumpmap)
edgeposition_stack = (edgeposition_stack - np.nanmean(edgeposition_stack))/np.nanstd(edgeposition_stack)
peak_stack = (peak_stack - np.nanmean(peak_stack))/np.nanstd(peak_stack)
goodness_of_fit = (goodness_of_fit - np.nanmean(goodness_of_fit))/np.nanstd(goodness_of_fit)
noisemap_stack = (noisemap_stack - np.nanmean(noisemap_stack))/np.nanstd(noisemap_stack)
peak_height = (peak_height - np.nanmean(peak_height))/np.nanstd(peak_height)
##########################

# # for debugging
# print np.count_nonzero(~np.isnan(edgejumpmap))
# print np.count_nonzero(~np.isnan(edgeposition_stack))
# print np.count_nonzero(~np.isnan(peak_stack))
# print np.count_nonzero(~np.isnan(goodness_of_fit))
# print np.count_nonzero(~np.isnan(noisemap_stack))
# print np.count_nonzero(~np.isnan(peak_height))
# print edgejumpmap.shape, edgeposition_stack.shape, peak_stack.shape, goodness_of_fit.shape, noisemap_stack.shape, peak_height.shape

##########################
# prepare matrix for clustering
# incoproate all features in to a box
features = []
if edgejumpmap_mode == 'on':
    features += list([edgejumpmap])
if edgeposition_stack_mode == 'on':
    features += list([edgeposition_stack])
if peak_stack_mode == 'on':
    features += list([peak_stack])
if goodness_of_fit_mode == 'on':
    features += list([goodness_of_fit])
if noisemap_stack_mode == 'on':
    features += list([noisemap_stack])
if peak_height_mode == 'on':
    features += list([peak_height])

features = np.array(features)
print features.shape
features = features.T
features = np.nan_to_num(features)
##########################

##########################
# clustering
# # # DBSCAN
# # db = DBSCAN(eps = 0.8, min_samples= 3)
# # labels = db.fit_predict(features)
#
# agglomerative clustering
ac = AgglomerativeClustering(n_clusters = 6)
labels = ac.fit_predict(features)

# # spectral clustering
# sc = SpectralClustering(n_clusters = 4)
# labels = sc.fit_predict(features)
###########################



# reshape the 1D array into 2 images
label1 = labels[:len(labels)/2].reshape(s1 ,s2)
label2 = labels[len(labels)/2:].reshape(s1 ,s2)

# visualization

save_path = '..\\report\\clustering\\'
plt.figure(1)
plt.title('clustering')
plt.imshow(label1, cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering1.png'))

plt.figure(2)
plt.title('clustering')
plt.imshow(label2, cmap = 'viridis')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering2.png'))