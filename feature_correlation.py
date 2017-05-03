"""
author: Fang Ren (SSRL)

5/2/2017
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import medfilt

feature_path = 'data\\features\\'

# import features
edgejumpmap = loadmat(feature_path+ 'edgejumpmap.mat')['edgejumpmap']
edgeposition_stack = loadmat(feature_path+ 'edgeposition_stack.mat')['edgeposition_stack']
peak_stack = loadmat(feature_path+ 'peak_stack.mat')['peak_stack']
Goodness_of_fit = loadmat(feature_path+ 'Goodness_of_fit.mat')['R_squares_stack']
noisemap_stack = loadmat(feature_path+ 'noisemap_stack.mat')['noisemap_stack']

# data cleaning
Goodness_of_fit = medfilt(Goodness_of_fit, kernel_size= 5) # filter some of the outliers

# get the dimension of the image
s1 = edgejumpmap.shape[0]
s2 = edgejumpmap.shape[1]

# flatten image
edgejumpmap = list(edgejumpmap.reshape(s1*s2, 1))
edgeposition_stack = list(edgeposition_stack.reshape(s1*s2, 1))
peak_stack = list(peak_stack.reshape(s1*s2, 1))
Goodness_of_fit = list(Goodness_of_fit.reshape(s1*s2, 1))
noisemap_stack = list(noisemap_stack.reshape(s1*s2, 1))
data = pd.DataFrame({'edge jump': edgejumpmap, 'edge position':edgeposition_stack, 'peak position':peak_stack, 'Goodness of fit':Goodness_of_fit, 'pre-edge noise':noisemap_stack})
data = data.astype(float)
names = list(data)
corr = data.corr()


# visualization
save_path = 'report\\variable_correlation'
if not os.path.exists(save_path):
    os.mkdir(save_path)
plt.matshow(abs(corr), cmap = 'viridis')
plt.xticks(range(len(corr.columns)), corr.columns, rotation = 70)
plt.yticks(range(len(corr.columns)), corr.columns, rotation = 40)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'correlation'))
plt.close('all')

# sns.jointplot(data.ix[:100000,0], data.ix[:100000,1], kind = 'kde')
# plt.savefig(os.path.join(save_path, 'test'))
# plt.close('all')

for i in range(len(names)):
    for j in range(i+1, len(names)):
        sns.jointplot(data[names[i]], data[names[j]], kind = 'kde')
        plt.savefig(os.path.join(save_path, names[i] + ' Vs ' + names[j]))
        plt.close('all')


