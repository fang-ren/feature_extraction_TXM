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

feature_path = 'data\\particle1\\'

# import features
edgejumpmap = loadmat(feature_path+ 'edgejumpmap.mat')['edgejumpmap']
edgeposition_stack = loadmat(feature_path+ 'edgeposition_stack.mat')['edgeposition_stack']
peak_stack = loadmat(feature_path+ 'peak_stack.mat')['peak_stack']
Goodness_of_fit = loadmat(feature_path+ 'R_squares_stack.mat')['R_squares_stack']
noisemap_stack = loadmat(feature_path+ 'noisemap_stack.mat')['noisemap_stack']
peak_height = loadmat(feature_path+ 'peakheight_stack.mat')['peakheight_stack']

print 'applying mask'
# applying mask, when the value is zero, set it to nan
edgejumpmap[(edgejumpmap == 0)] = np.nan
edgeposition_stack[edgeposition_stack == 0] = np.nan
peak_stack[peak_stack == 0] = np.nan
Goodness_of_fit[Goodness_of_fit == 0] = np.nan
noisemap_stack[noisemap_stack == 0] = np.nan
peak_height[peak_height == -1] = np.nan

print 'save GOF'
np.savetxt('GoF.csv', Goodness_of_fit, delimiter=',')

print 'removing outliers'
# remove outliers using 3 sigma, replace it with nan
edgejumpmap[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
edgeposition_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
peak_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
Goodness_of_fit[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
noisemap_stack[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan
peak_height[abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)] = np.nan

edgejumpmap[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
edgeposition_stack[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
peak_stack[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
Goodness_of_fit[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
noisemap_stack[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan
peak_height[abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)] = np.nan


edgejumpmap[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
edgeposition_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
peak_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
noisemap_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
peak_height[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan

edgejumpmap[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
edgeposition_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
peak_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
Goodness_of_fit[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
noisemap_stack[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan
peak_height[abs(Goodness_of_fit - np.nanmean(Goodness_of_fit)) > 3 * np.nanstd(Goodness_of_fit)] = np.nan

print np.nanmedian(Goodness_of_fit)

# get the dimension of the image
s1 = edgejumpmap.shape[0]
s2 = edgejumpmap.shape[1]

print 'flattening image'
# flatten image
edgejumpmap = list(edgejumpmap.reshape(s1*s2, 1)[:,0])
edgeposition_stack = list(edgeposition_stack.reshape(s1*s2, 1)[:,0])
peak_stack = list(peak_stack.reshape(s1*s2, 1)[:,0])
Goodness_of_fit = list(Goodness_of_fit.reshape(s1*s2, 1)[:,0])
noisemap_stack = list(noisemap_stack.reshape(s1*s2, 1)[:,0])
peak_height = list(peak_height.reshape(s1*s2, 1)[:,0])

data = pd.DataFrame({'edge jump': edgejumpmap, 'edge position':edgeposition_stack, 'peak position':peak_stack, 'Goodness of fit':Goodness_of_fit, 'pre-edge noise':noisemap_stack, 'peak height':peak_height})
data = data.astype(float)

print 'visualization'
# visualization
save_path = 'report\\variable_correlation'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# print Goodness_of_fit
sns.kdeplot(np.nan_to_num(np.array(Goodness_of_fit)))
plt.xlim(0, 120)
plt.savefig(os.path.join(save_path, 'Goodness_of_fit'))

names = list(data)
corr = data.corr()
print corr



plt.matshow(abs(corr).T, cmap = 'viridis')
plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
plt.yticks(range(len(corr.columns)), corr.columns, rotation = 50)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'correlation'))
plt.close('all')


sns.set(font_scale=1.5)
sns.jointplot(data['Goodness of fit'], data['edge jump'], kind = 'scatter', xlim = (-0.12, 1), ylim = (-0.05, 0.6), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Goodness of fit' + ' Vs ' + 'edge jump'))
plt.close('all')

sns.jointplot(data['Goodness of fit'], data['edge position'], kind = 'scatter', xlim = (-0.12, 1), ylim = (7680, 7770), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Goodness of fit' + ' Vs ' + 'edge position'))
plt.close('all')

sns.jointplot(data['Goodness of fit'], data['peak height'], kind = 'scatter', xlim = (-0.12, 1), ylim = (-0.5, 3), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Goodness of fit' + ' Vs ' + 'peak height'))
plt.close('all')


sns.jointplot(data['Goodness of fit'], data['peak position'], kind = 'scatter', xlim = (-0.12, 1), ylim = (7725, 7737), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Goodness of fit' + ' Vs ' + 'peak position'))
plt.close('all')

sns.jointplot(data['Goodness of fit'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.12, 1), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Goodness of fit' + ' Vs ' + 'pre-edge noise'))
plt.close('all')


#

sns.jointplot(data['edge jump'], data['edge position'], kind = 'scatter', xlim = (-0.05, 0.6), ylim = (7680, 7770), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'edge position'))
plt.close('all')

sns.jointplot(data['edge jump'], data['peak height'], kind = 'scatter', xlim = (-0.05, 0.6), ylim = (-0.5, 3), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak height'))
plt.close('all')

sns.jointplot(data['edge jump'], data['peak position'], kind = 'scatter', xlim = (-0.05, 0.6), ylim = (7725, 7737), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak position'))
plt.close('all')

sns.jointplot(data['edge jump'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.05, 0.6), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'pre-edge noise'))
plt.close('all')



#
sns.jointplot(data['edge position'], data['peak height'], kind = 'scatter', xlim = (7680, 7770), ylim = (-0.5, 3), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak height'))
plt.close('all')

sns.jointplot(data['edge position'], data['peak position'], kind = 'scatter', xlim = (7680, 7770), ylim = (7725, 7737), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak position'))
plt.close('all')

sns.jointplot(data['edge position'], data['pre-edge noise'], kind = 'scatter', xlim = (7680, 7770), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'pre-edge noise'))
plt.close('all')


#


sns.jointplot(data['peak height'], data['peak position'], kind = 'scatter', xlim = (-0.5, 3), ylim = (7725, 7737), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'peak position'))
plt.close('all')

sns.jointplot(data['peak height'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.5, 3), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'pre-edge noise'))
plt.close('all')

#

sns.jointplot(data['peak position'], data['pre-edge noise'], kind = 'scatter', xlim = (7725, 7737), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'peak position' + ' Vs ' + 'pre-edge noise'))
plt.close('all')
