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
from sklearn.cluster import KMeans
from import_features import import_features
from compress_image import compress_image
from zero_mask import zero_mask
from odd_value_mask import odd_value_mask
from outlier_mask import outlier_mask
from flatten_imArray import flatten_imArray
from circular_mask import circular_mask
from scipy.io import savemat

#########################
# user input
edgejumpmap_mode = 'off'
edgeposition_stack_mode = 'on'
peak_stack_mode = 'on'
goodness_of_fit_mode = 'on'
noisemap_stack_mode = 'on'
peak_height_mode = 'on'
weight = [0.1, # edge jump
          1,   # edge position
          1,   # peak position
          0.1,   # goodness_of_fit
          0.1,   # pre-edge noise
          1]   # peak height

num_clusters = 4

clustering_method = 'KM'
#########################

# #########################
# # import cell_1
# feature_path1 = '..\\data\\100_particle_data\cell_1\A\\'
# feature_path2 = '..\\data\\100_particle_data\cell_1\B\\'
# feature_path3 = '..\\data\\100_particle_data\cell_1\C\\'
# feature_path4 = '..\\data\\100_particle_data\cell_1\D\\'
# feature_path5 = '..\\data\\100_particle_data\cell_1\E\\'
# feature_path6 = '..\\data\\100_particle_data\cell_1\F\\'
# feature_path7 = '..\\data\\100_particle_data\cell_1\G\\'
# feature_path8 = '..\\data\\100_particle_data\cell_1\H\\'
# feature_paths = [feature_path1, feature_path2, feature_path3, feature_path4, feature_path5, feature_path6, feature_path7, feature_path8]
# names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# save_path = '..\\data\\100_particle_data\\clustering\\cell_1\\'

# ###########################
# # import cell_2
# feature_path9 = '..\\data\\100_particle_data\cell_2\A\\'
# feature_path10 = '..\\data\\100_particle_data\cell_2\C\\'
# feature_path11 = '..\\data\\100_particle_data\cell_2\D\\'
# feature_path12 = '..\\data\\100_particle_data\cell_2\E\\'
# feature_path13 = '..\\data\\100_particle_data\cell_2\F\\'
# feature_path14 = '..\\data\\100_particle_data\cell_2\G\\'
# feature_paths = [feature_path9, feature_path10, feature_path11, feature_path12, feature_path13, feature_path14]
# names = ['A', 'C', 'D', 'E', 'F', 'G']
# save_path = '..\\data\\100_particle_data\\clustering\\cell_2\\'
#

# import 2 cells at a time
# import cell_1
feature_path1 = '..\\data\\100_particle_data\cell_1\A\\'
feature_path2 = '..\\data\\100_particle_data\cell_1\B\\'
feature_path3 = '..\\data\\100_particle_data\cell_1\C\\'
feature_path4 = '..\\data\\100_particle_data\cell_1\D\\'
feature_path5 = '..\\data\\100_particle_data\cell_1\E\\'
feature_path6 = '..\\data\\100_particle_data\cell_1\F\\'
feature_path7 = '..\\data\\100_particle_data\cell_1\G\\'
feature_path8 = '..\\data\\100_particle_data\cell_1\H\\'

# import cell_2
feature_path9 = '..\\data\\100_particle_data\cell_2\A\\'
feature_path10 = '..\\data\\100_particle_data\cell_2\C\\'
feature_path11 = '..\\data\\100_particle_data\cell_2\D\\'
feature_path12 = '..\\data\\100_particle_data\cell_2\E\\'
feature_path13 = '..\\data\\100_particle_data\cell_2\F\\'
feature_path14 = '..\\data\\100_particle_data\cell_2\G\\'
feature_paths = [feature_path1, feature_path2, feature_path3, feature_path4, feature_path5, feature_path6, feature_path7, feature_path8, feature_path9, feature_path10, feature_path11, feature_path12, feature_path13, feature_path14]
names = ['1_A', '1_B', '1_C', '1_D', '1_E', '1_F', '1_G', '1_H', '2_A', '2_C', '2_D', '2_E', '2_F', '2_G']
save_path = '..\\data\\100_particle_data\\clustering\\'


edgejumpmaps = []
edgeposition_stacks = []
peak_stacks = []
goodness_of_fits = []
noisemap_stacks = []
peak_heights = []
dim1s = []
dim2s = []


for feature_path in feature_paths:
    # import features
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = import_features(feature_path)
    
    # compress features to speed up
    # edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = compress_image(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
    
    # remove odd values in the features
    # edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = odd_value_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
    
    # mask data points with zero with nan
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = zero_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
    
    # # applying outlier mask for both particles
    # edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = outlier_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
    
    # flatten images
    dim1 = edgejumpmap.shape[0]
    dim2 = edgejumpmap.shape[1]
    dim1s.append(dim1)
    dim2s.append(dim2)
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = flatten_imArray(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)

    # add the current particle features
    edgejumpmaps += edgejumpmap
    edgeposition_stacks += edgeposition_stack
    peak_stacks += peak_stack
    goodness_of_fits += goodness_of_fit
    noisemap_stacks += noisemap_stack
    peak_heights += peak_height

edgejumpmaps = np.array(edgejumpmaps)
edgeposition_stacks = np.array(edgeposition_stacks)
peak_stacks = np.array(peak_stacks)
goodness_of_fits = np.array(goodness_of_fits)
noisemap_stacks = np.array(noisemap_stacks)
peak_heights = np.array(peak_heights)
#print edgejumpmaps.shape, edgeposition_stacks.shape, peak_stacks.shape, goodness_of_fits.shape, noisemap_stacks.shape, peak_heights.shape


##########################
# standardized features
edgejumpmaps = (edgejumpmaps - np.nanmean(edgejumpmaps))/np.nanstd(edgejumpmaps)
edgeposition_stacks = (edgeposition_stacks - np.nanmean(edgeposition_stacks))/np.nanstd(edgeposition_stacks)
peak_stacks = (peak_stacks - np.nanmean(peak_stacks))/np.nanstd(peak_stacks)
goodness_of_fits = (goodness_of_fits - np.nanmean(goodness_of_fits))/np.nanstd(goodness_of_fits)
noisemap_stacks = (noisemap_stacks - np.nanmean(noisemap_stacks))/np.nanstd(noisemap_stacks)
peak_heights = (peak_heights - np.nanmean(peak_heights))/np.nanstd(peak_heights)
##########################


##########################
# weigh features
edgejumpmaps = edgejumpmaps * weight[0]
edgeposition_stacks = edgeposition_stacks * weight[1]
peak_stacks = peak_stacks * weight[2]
goodness_of_fits = goodness_of_fits * weight[3]
noisemap_stacks = noisemap_stacks * weight[4]
peak_heights = peak_heights * weight[5]
##########################


# # for debugging
# print np.count_nonzero(~np.isnan(edgejumpmap))
# print np.count_nonzero(~np.isnan(edgeposition_stacks))
# print np.count_nonzero(~np.isnan(peak_stacks))
# print np.count_nonzero(~np.isnan(goodness_of_fits))
# print np.count_nonzero(~np.isnan(noisemap_stacks))
# print np.count_nonzero(~np.isnan(peak_heights))
# print edgejumpmap.shape, edgeposition_stacks.shape, peak_stacks.shape, goodness_of_fits.shape, noisemap_stacks.shape, peak_heights.shape

##########################
# prepare matrix for clustering
# incoproate all features in to a box
features = []
if edgejumpmap_mode == 'on':
    features += list([edgejumpmaps])
if edgeposition_stack_mode == 'on':
    features += list([edgeposition_stacks])
if peak_stack_mode == 'on':
    features += list([peak_stacks])
if goodness_of_fit_mode == 'on':
    features += list([goodness_of_fits])
if noisemap_stack_mode == 'on':
    features += list([noisemap_stacks])
if peak_height_mode == 'on':
    features += list([peak_heights])

features = np.array(features)
print features.shape
features = features.T

nan_mask = np.isnan(features[:,0])
print nan_mask.shape
features = np.nan_to_num(features)
##########################


##########################
## clustering
# # DBSCAN
# db = DBSCAN(eps = 0.5, min_samples= num_clusters)
# labels = db.fit_predict(features)

if clustering_method == 'AC':
    # agglomerative clustering
    ac = AgglomerativeClustering(n_clusters = num_clusters, linkage= 'ward')
    labels = ac.fit_predict(features)

elif clustering_method == 'SC':
    # spectral clustering
    sc = SpectralClustering(n_clusters = num_clusters)
    labels = sc.fit_predict(features)

elif clustering_method == 'KM':
    # spectral clustering
    km = KMeans(n_clusters = num_clusters, random_state= 1)
    labels = km.fit_predict(features)

##########################


# reshape the 1D array into 2 images
labels = labels.astype(float)
print labels.shape
# print labels[nan_mask]
labels[nan_mask] = np.nan


# visualization
start = 0
for i, name in enumerate(names[:len(feature_paths)]):
    #print i, name
    dim1 = dim1s[i]
    dim2 = dim2s[i]
    label = labels[start:(start+dim1*dim2)].reshape(dim1 ,dim2)
    start += dim1*dim2
    plt.figure(i+1)
    plt.title('particle '+ name)
    plt.imshow(label, cmap = 'viridis')
    plt.grid('off')
    plt.colorbar()
    plt.clim(0, 3)
    plt.savefig(os.path.join(save_path, 'clustering'+ name))
    savemat(save_path + 'particle' + name + '.mat', {'labels':label})
