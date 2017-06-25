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


#########################
# import 1st particle
feature_path1 = '..\\data\\particle1\\'
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = import_features(feature_path1)
peak_stack1 = peak_stack1 +1.5

#########################

#########################
# import 2nd particle
feature_path2 = '..\\data\\particle3\\'
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = import_features(feature_path2)
peak_stack2 = peak_stack2 -0.15
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
# flatten images
s1 = edgejumpmap1.shape[0]
s2 = edgejumpmap1.shape[1]
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################

peak_stack1 = np.array(peak_stack1)
peak_stack2 = np.array(peak_stack2)

plt.figure(1)
plt.subplot(211)
y,binEdges=np.histogram(peak_stack1[~np.isnan(peak_stack1)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 1')
plt.xlim(7730, 7734)
plt.legend()
plt.subplot(212)
y,binEdges=np.histogram(peak_stack2[~np.isnan(peak_stack2)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 2')
plt.xlim(7730, 7734)
plt.legend()
plt.savefig('peak position.png')

edgeposition_stack1 = np.array(edgeposition_stack1)
edgeposition_stack2 = np.array(edgeposition_stack2)
plt.figure(2)
plt.subplot(211)
y,binEdges=np.histogram(edgeposition_stack1[~np.isnan(edgeposition_stack1)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 1')
plt.xlim(7710, 7725)
plt.ylim(0, 200)
plt.legend()
plt.subplot(212)
y,binEdges=np.histogram(edgeposition_stack2[~np.isnan(edgeposition_stack2)],bins=1000)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
plt.plot(bincenters,y,'-', label = 'particle 2')
plt.xlim(7710, 7725)
plt.ylim(0, 200)
plt.legend()
plt.savefig('edge position.png')