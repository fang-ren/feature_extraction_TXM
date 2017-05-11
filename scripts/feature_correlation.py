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

feature_path = 'data\\features_2\\'

# import features
edgejumpmap = loadmat(feature_path+ 'edgejumpmap.mat')['edgejumpmap']
edgeposition_stack = loadmat(feature_path+ 'edgeposition_stack.mat')['edgeposition_stack']
peak_stack = loadmat(feature_path+ 'peak_stack.mat')['peak_stack']
Goodness_of_fit = loadmat(feature_path+ 'Goodness_of_fit.mat')['R_squares_stack']
noisemap_stack = loadmat(feature_path+ 'noisemap_stack.mat')['noisemap_stack']


# applying mask, when the value is zero, set it to nan
edgejumpmap[(edgejumpmap == 0)] = np.nan
edgeposition_stack[edgeposition_stack == 0 * (abs(edgeposition_stack - np.mean(edgeposition_stack)) > 3 * np.std(edgeposition_stack))] = np.nan
peak_stack[peak_stack == 0] = np.nan
Goodness_of_fit[Goodness_of_fit == 0] = np.nan
noisemap_stack[noisemap_stack == 0] = np.nan

# remove outliers using 3 sigma, replace it with nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
peak_stack[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan

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

print names

sns.jointplot(data[names[0]], data[names[1]], kind = 'kde', )
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[0] + ' Vs ' + names[1]))
plt.close('all')


sns.jointplot(data[names[0]], data[names[2]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[0] + ' Vs ' + names[2]))
plt.close('all')


sns.jointplot(data[names[0]], data[names[3]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[0] + ' Vs ' + names[3]))
plt.close('all')


sns.jointplot(data[names[0]], data[names[4]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[0] + ' Vs ' + names[4]))
plt.close('all')


sns.jointplot(data[names[1]], data[names[2]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[1] + ' Vs ' + names[2]))
plt.close('all')


sns.jointplot(data[names[1]], data[names[3]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[1] + ' Vs ' + names[3]))
plt.close('all')


sns.jointplot(data[names[1]], data[names[4]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[1] + ' Vs ' + names[4]))
plt.close('all')


sns.jointplot(data[names[2]], data[names[3]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[2] + ' Vs ' + names[3]))
plt.close('all')

sns.jointplot(data[names[2]], data[names[4]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[2] + ' Vs ' + names[4]))
plt.close('all')


sns.jointplot(data[names[3]], data[names[4]], kind = 'kde')
plt.tight_layout()
plt.savefig(os.path.join(save_path, names[3] + ' Vs ' + names[4]))
plt.close('all')
