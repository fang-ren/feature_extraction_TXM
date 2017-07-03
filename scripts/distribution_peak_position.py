"""
author: Fang Ren (SSRL)

5/24/2017
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
from outlier_mask import outlier_mask
from flatten_imArray import flatten_imArray
from circular_mask import circular_mask


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
# import 3rd particle
feature_path3 = '..\\data\\particle3\\'
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = import_features(feature_path3)
peak_stack3[peak_stack3 == -1.2] = 0
#########################

# elem, count = np.unique(peak_stack3, return_counts= True)
# # print np.concatenate(elem, count)
# np.savetxt('count.csv', np.concatenate(([elem], [count])).T, delimiter=',')

#########################
# applying circular mask for both particles
mask1 = circular_mask((edgejumpmap1.shape[0],edgejumpmap1.shape[1]), (195, 227), 46)
print mask1
edgejumpmap1[mask1] = np.nan
edgeposition_stack1[mask1] = np.nan
goodness_of_fit1[mask1] = np.nan
peak_height1[mask1] = np.nan
peak_stack1[mask1] = np.nan

mask2 = circular_mask((edgejumpmap1.shape[0],edgejumpmap1.shape[1]), (170, 220), 120)
edgejumpmap2[mask2] = np.nan
edgeposition_stack2[mask2] = np.nan
goodness_of_fit2[mask2] = np.nan
peak_height2[mask2] = np.nan
peak_stack2[mask2] = np.nan
noisemap_stack2[mask2] = np.nan
#########################

#########################
# applying zero mask for both particles
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = zero_mask(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = zero_mask(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = zero_mask(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################


peak_stack1 = peak_stack1 +1.5
edgeposition_stack1 = edgeposition_stack1
peak_stack3 = peak_stack3
edgeposition_stack3 = edgeposition_stack3
#
# plt.imshow(peak_stack3, cmap = 'viridis')
# plt.clim(7730, 7734)
# plt.savefig('1')
#
# plt.imshow(peak_stack3, cmap = 'viridis')
# plt.clim(7730, 7734)
# plt.savefig('2')

#########################
# flatten images
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = flatten_imArray(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################

# plot peak positions
peak_stack1 = np.array(peak_stack1)
peak_stack2 = np.array(peak_stack2)
peak_stack3 = np.array(peak_stack3)

plt.figure(1)
plt.subplot(311)
y,binEdges=np.histogram(peak_stack1[~np.isnan(peak_stack1)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 1')
plt.plot([7731] * 10, range(0, 300, 30), 'y')
plt.plot([7732] * 10, range(0, 300, 30), 'darkblue')
plt.plot([7728] * 10, range(0, 300, 30), 'g')
plt.plot([7730] * 10, range(0, 300, 30), 'steelblue')
plt.xlim(7727, 7734)
plt.legend()
plt.subplot(312)
y,binEdges=np.histogram(peak_stack2[~np.isnan(peak_stack2)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot([7731] * 10, range(0, 500, 50), 'y')
plt.plot([7732] * 10, range(0, 500, 50), 'darkblue')
plt.plot([7728] * 10, range(0, 500, 50), 'g')
plt.plot([7730] * 10, range(0, 500, 50), 'steelblue')
plt.plot(bincenters,y,'-', label = 'particle 2')
plt.xlim(7727, 7734)
plt.legend()
plt.subplot(313)
y,binEdges=np.histogram(peak_stack3[~np.isnan(peak_stack3)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot([7731] * 10, range(0, 350, 35), 'y')
plt.plot([7732] * 10, range(0, 350, 35), 'darkblue')
plt.plot([7728] * 10, range(0, 350, 35), 'g')
plt.plot([7730] * 10, range(0, 350, 35), 'steelblue')
plt.plot(bincenters,y,'-', label = 'particle 3')
plt.xlim(7727, 7734)
# plt.ylim(0, 600)
plt.legend()
plt.savefig('peak position.png')

# plot edgeposition_stack
edgeposition_stack1 = np.array(edgeposition_stack1)
edgeposition_stack2 = np.array(edgeposition_stack2)
edgeposition_stack3 = np.array(edgeposition_stack3)
plt.figure(2)
plt.subplot(311)
y,binEdges=np.histogram(edgeposition_stack1[~np.isnan(edgeposition_stack1)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 1')
plt.plot([7717.95] * 10, range(0, 200, 20), 'y')
plt.plot([7722.5] * 10, range(0, 200, 20), 'darkblue')
plt.plot([7718] * 10, range(0, 200, 20), 'g')
plt.plot([7722] * 10, range(0, 200, 20), 'steelblue')
plt.xlim(7714, 7724)
plt.ylim(0, 200)
plt.legend()
plt.subplot(312)
y,binEdges=np.histogram(edgeposition_stack2[~np.isnan(edgeposition_stack2)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 2')
plt.plot([7717.95] * 10, range(0, 200, 20), 'y')
plt.plot([7722.5] * 10, range(0, 200, 20), 'darkblue')
plt.plot([7718] * 10, range(0, 200, 20), 'g')
plt.plot([7722] * 10, range(0, 200, 20), 'steelblue')
plt.xlim(7714, 7724)
plt.ylim(0, 300)
plt.legend()
plt.subplot(313)
y,binEdges=np.histogram(edgeposition_stack3[~np.isnan(edgeposition_stack3)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 3')
plt.plot([7717.95] * 10, range(0, 200, 20), 'y')
plt.plot([7722.5] * 10, range(0, 200, 20), 'darkblue')
plt.plot([7718] * 10, range(0, 200, 20), 'g')
plt.plot([7722] * 10, range(0, 200, 20), 'steelblue')
plt.xlim(7714, 7724)
plt.ylim(0, 500)
plt.legend()
plt.savefig('edge position.png')