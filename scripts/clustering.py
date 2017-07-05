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

#########################
# import 1st particle
feature_path1 = '..\\data\\particle1\\'
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = import_features(feature_path1)
edgeposition_stack1 = edgeposition_stack1
#########################

#########################
# import 2nd particle
feature_path2 = '..\\data\\particle2\\'
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = import_features(feature_path2)
#########################
#
#########################
# import 3rd particle
feature_path3 = 'C:\\Research_FangRen\\Python codes\\feature_extraction_TXM\\data\\particle3\\'
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = import_features(feature_path3)
#########################
#
# #########################
# # compress images for speeding up, uncomment for real results
# edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = compress_image(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
# edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = compress_image(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
# #########################

# #########################
# # applying circular mask for both particles
# mask1 = circular_mask((edgejumpmap1.shape[0],edgejumpmap1.shape[1]), (195, 227), 46)
# print mask1
# edgejumpmap1[mask1] = np.nan
# edgeposition_stack1[mask1] = np.nan
# goodness_of_fit1[mask1] = np.nan
# peak_height1[mask1] = np.nan
# peak_stack1[mask1] = np.nan
#
# mask2 = circular_mask((edgejumpmap1.shape[0],edgejumpmap1.shape[1]), (170, 220), 120)
# edgejumpmap2[mask2] = np.nan
# edgeposition_stack2[mask2] = np.nan
# goodness_of_fit2[mask2] = np.nan
# peak_height2[mask2] = np.nan
# peak_stack2[mask2] = np.nan
# noisemap_stack2[mask2] = np.nan
# #########################

#########################
# applying odd value mask for  particles
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = odd_value_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = odd_value_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = odd_value_mask(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################

#########################
# applying zero mask for both particles
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = zero_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = zero_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = zero_mask(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################

# #########################
# # applying outlier mask for both particles
# edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = outlier_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
# edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = outlier_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
# #########################

#########################
# visualizing features
save_path = '..\\report\\feature_visualization\\'
# particle 1
plt.figure(1)
plt.title('edge jump')
plt.imshow(np.nan_to_num(edgejumpmap1), cmap = 'jet')
plt.grid('off')
plt.clim(0.01, 0.65)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge jump_particle1.png'))
plt.figure(2)
plt.title('edge position')
plt.imshow(np.nan_to_num(edgeposition_stack1), cmap = 'jet')
plt.grid('off')
plt.clim(7710, 7725)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge position_particle1.png'))
plt.figure(3)
plt.title('first peak position')
plt.imshow(np.nan_to_num(peak_stack1), cmap = 'jet')
plt.grid('off')
plt.clim(7724, 7738)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'first peak position_particle1.png'))
plt.figure(4)
plt.title('goodness of fit')
plt.imshow(np.nan_to_num(goodness_of_fit1), cmap = 'jet')
plt.grid('off')
plt.clim(0, 0.18)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'goodness of fit_particle1.png'))
plt.figure(5)
plt.title('pre-edge noise')
plt.imshow(np.nan_to_num(noisemap_stack1), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(0.01, 0.04)
plt.savefig(os.path.join(save_path, 'pre-edge noise_particle1.png'))
plt.figure(6)
plt.title('peak height')
plt.imshow(np.nan_to_num(peak_height1), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(-0.3, 3)
plt.savefig(os.path.join(save_path, 'peak height_particle1.png'))
plt.close('all')

# particle 2
plt.figure(7)
plt.title('edge jump')
plt.imshow(np.nan_to_num(edgejumpmap2), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(0.01, 0.65)
plt.savefig(os.path.join(save_path, 'edge jump_particle2.png'))
plt.figure(8)
plt.title('edge position')
plt.imshow(np.nan_to_num(edgeposition_stack2), cmap = 'jet')
plt.grid('off')
plt.clim(7710, 7725)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge position_particle2.png'))
plt.figure(9)
plt.title('first peak position')
plt.imshow(np.nan_to_num(peak_stack2), cmap = 'jet')
plt.grid('off')
plt.clim(7724, 7738)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'first peak position_particle2.png'))
plt.figure(10)
plt.title('goodness of fit')
plt.imshow(np.nan_to_num(goodness_of_fit2), cmap = 'jet')
plt.grid('off')
plt.clim(0, 0.18)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'goodness of fit_particle2.png'))
plt.figure(11)
plt.title('pre-edge noise')
plt.imshow(np.nan_to_num(noisemap_stack2), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(0.01, 0.04)
plt.savefig(os.path.join(save_path, 'pre-edge noise_particle2.png'))
plt.figure(12)
plt.title('peak height')
plt.imshow(np.nan_to_num(peak_height2), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(-0.3, 3)
plt.savefig(os.path.join(save_path, 'peak height_particle2.png'))

plt.close('all')
########################


# particle 3
plt.figure(7)
plt.title('edge jump')
plt.imshow(np.nan_to_num(edgejumpmap3), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(0.01, 0.65)
plt.savefig(os.path.join(save_path, 'edge jump_particle3.png'))
plt.figure(8)
plt.title('edge position')
plt.imshow(np.nan_to_num(edgeposition_stack3), cmap = 'jet')
plt.grid('off')
plt.clim(7710, 7725)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'edge position_particle3.png'))
plt.figure(9)
plt.title('first peak position')
plt.imshow(np.nan_to_num(peak_stack3), cmap = 'jet')
plt.grid('off')
plt.clim(7724, 7738)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'first peak position_particle3.png'))
plt.figure(10)
plt.title('goodness of fit')
plt.imshow(np.nan_to_num(goodness_of_fit3), cmap = 'jet')
plt.grid('off')
plt.clim(0, 0.18)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'goodness of fit_particle3.png'))
plt.figure(11)
plt.title('pre-edge noise')
plt.imshow(np.nan_to_num(noisemap_stack3), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(0.01, 0.04)
plt.savefig(os.path.join(save_path, 'pre-edge noise_particle3.png'))
plt.figure(12)
plt.title('peak height')
plt.imshow(np.nan_to_num(peak_height3), cmap = 'jet')
plt.grid('off')
plt.colorbar()
plt.clim(-0.3, 3)
plt.savefig(os.path.join(save_path, 'peak height_particle3.png'))

plt.close('all')
########################

#########################
# flatten images
s1_1 = edgejumpmap1.shape[0]
s2_1 = edgejumpmap1.shape[1]
s1_2 = edgejumpmap2.shape[0]
s2_2 = edgejumpmap2.shape[1]
s1_3 = edgejumpmap3.shape[0]
s2_3 = edgejumpmap3.shape[1]
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = flatten_imArray(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################

#########################
# conbine two particle data
edgejumpmap = np.concatenate((edgejumpmap1,edgejumpmap2, edgejumpmap3))
edgeposition_stack = np.concatenate((edgeposition_stack1, edgeposition_stack2, edgeposition_stack3))
peak_stack = np.concatenate((peak_stack1, peak_stack2, peak_stack3))
goodness_of_fit = np.concatenate((goodness_of_fit1, goodness_of_fit2, goodness_of_fit3))
noisemap_stack = np.concatenate((noisemap_stack1, noisemap_stack2, noisemap_stack3))
peak_height = np.concatenate((peak_height1, peak_height2, peak_height3))
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


##########################
# weigh features
edgejumpmap = edgejumpmap * weight[0]
edgeposition_stack = edgeposition_stack * weight[1]
peak_stack = peak_stack * weight[2]
goodness_of_fit = goodness_of_fit * weight[3]
noisemap_stack = noisemap_stack * weight[4]
peak_height = peak_height * weight[5]
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
print labels
print labels[nan_mask]
labels[nan_mask] = np.nan

print labels.shape
print s1_1,s2_1, s1_2, s2_2, s1_3,s2_3

label1 = labels[:s1_1*s2_1].reshape(s1_1 ,s2_1)
label2 = labels[s1_1*s2_1:(s1_2*s2_2+s1_1*s2_1)].reshape(s1_2 ,s2_2)
label3 = labels[(s1_2*s2_2+s1_1*s2_1):].reshape(s1_3 ,s2_3)


# visualization
save_path = '..\\report\\clustering\\'
plt.figure(1)
plt.title('particle 1')
plt.imshow(label1, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering1.png'))

plt.figure(2)
plt.title('particle 2')
plt.imshow(label2, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering2.png'))

plt.figure(3)
plt.title('particle 3')
plt.imshow(label3, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering3.png'))

savemat(save_path + 'particle1_label.mat', {'labels':label1})
savemat(save_path + 'particle2_label.mat', {'labels':label2})
savemat(save_path + 'particle3_label.mat', {'labels':label3})

# np.savetxt(save_path+'particle1_label.csv', label1, delimiter=',')
# np.savetxt(save_path+'particle2_label.csv', label2, delimiter=',')
# np.savetxt(save_path+'particle3_label.csv', label2, delimiter=',')