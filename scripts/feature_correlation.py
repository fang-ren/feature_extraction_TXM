"""
author: Fang Ren (SSRL)

5/2/2017
"""

import matplotlib.pyplot as plt
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
from import_features import import_features
from zero_mask import zero_mask
from outlier_mask import outlier_mask
from flatten_imArray import flatten_imArray

#########################
# import 1st particle
feature_path1 = 'data\\particle1\\'
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = import_features(feature_path1)
#########################

#########################
# import 2nd particle
feature_path2 = 'data\\particle2\\'
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = import_features(feature_path2)
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
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
#########################


#########################
# conbine two particle data
edgejumpmap = edgejumpmap1 + edgejumpmap2
edgeposition_stack = edgeposition_stack1 + edgeposition_stack2
peak_stack = peak_stack1 + peak_stack2
goodness_of_fit = goodness_of_fit1 + goodness_of_fit2
noisemap_stack = noisemap_stack1 + noisemap_stack2
peak_height = peak_height1 + peak_height2

data = pd.DataFrame({'edge jump': edgejumpmap, 'edge position':edgeposition_stack, 'peak position':peak_stack, 'goodness of fit':goodness_of_fit, 'pre-edge noise':noisemap_stack, 'peak height':peak_height})
data = data.astype(float)
#########################

print np.count_nonzero(~np.isnan(edgejumpmap))
print np.count_nonzero(~np.isnan(edgeposition_stack))
print np.count_nonzero(~np.isnan(peak_stack))
print np.count_nonzero(~np.isnan(goodness_of_fit))
print np.count_nonzero(~np.isnan(noisemap_stack))
print np.count_nonzero(~np.isnan(peak_height))


print 'visualization'
# visualization
save_path = 'report\\variable_correlation'
if not os.path.exists(save_path):
    os.mkdir(save_path)
#
# # print goodness_of_fit
# sns.kdeplot(np.nan_to_num(np.array(goodness_of_fit)))
# plt.xlim(0, 120)
# plt.savefig(os.path.join(save_path, 'goodness_of_fit'))

names = list(data)
corr = data.corr()
print corr


plt.matshow(abs(corr).T, cmap = 'viridis')
plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
plt.yticks(range(len(corr.columns)), corr.columns, rotation = 50)
plt.colorbar()
plt.savefig(os.path.join(save_path, 'correlation'), dpi = 600)
plt.close('all')


sns.set(font_scale=1.7, style = 'white')

#
#sns.jointplot(data['edge jump'], data['edge position'], kind = 'scatter', xlim = (-0.05, 0.7), ylim = (7700, 7785), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['edge position'], kind = 'kde', xlim = (-0.05, 0.7), ylim = (7700, 7785), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "edge position")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'edge position'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge jump'], data['peak height'], kind = 'scatter', xlim = (-0.05, 0.7), ylim = (-1, 2.2), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['goodness of fit'], kind = 'kde', xlim = (-0.05, 0.7), ylim = (-0.5, 7), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "goodness of fit")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'goodness of fit'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge jump'], data['peak height'], kind = 'scatter', xlim = (-0.05, 0.7), ylim = (-1, 2.2), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['peak height'], kind = 'kde', xlim = (-0.05, 0.7), ylim = (-1, 2.2), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "peak height")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak height'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge jump'], data['peak position'], kind = 'scatter', xlim = (-0.05, 0.7), ylim = (7726, 7736), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['peak position'], kind = 'kde', xlim = (-0.05, 0.7), ylim = (7726, 7736), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "peak position")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge jump'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.05, 0.7), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['pre-edge noise'], kind = 'kde', xlim = (-0.05, 0.7), ylim = (0.006, 0.035), dropna = True, n_levels = 30)
plt.tight_layout()
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("edge jump", "pre-edge noise")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')



# sns.jointplot(data['edge position'], data['peak height'], kind = 'scatter', xlim = (7700, 7785), ylim = (-1, 2.2), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge position'], data['goodness of fit'], kind = 'kde', xlim = (7700, 7785), ylim = (-0.5, 7), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'goodness of fit'), dpi = 600)
plt.close('all')


# sns.jointplot(data['edge position'], data['peak height'], kind = 'scatter', xlim = (7700, 7785), ylim = (-1, 2.2), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge position'], data['peak height'], kind = 'kde', xlim = (7700, 7785), ylim = (-1, 2.2), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak height'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge position'], data['peak position'], kind = 'scatter', xlim = (7700, 7785), ylim = (7726, 7736), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge position'], data['peak position'], kind = 'kde', xlim = (7700, 7785), ylim = (7726, 7736), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')

# sns.jointplot(data['edge position'], data['pre-edge noise'], kind = 'scatter', xlim = (7700, 7785), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge position'], data['pre-edge noise'], kind = 'kde', xlim = (7700, 7785), ylim = (0.006, 0.035), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("edge position", "")
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')

#

# sns.jointplot(data['goodness of fit'], data['peak height'], kind = 'scatter', xlim = (-0.5, 7), ylim = (-1, 2.2), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['goodness of fit'], data['peak height'], kind = 'kde', xlim = (-0.5, 7), ylim = (-1, 2.2), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'peak height'), dpi = 600)
plt.close('all')
#

# sns.jointplot(data['goodness of fit'], data['peak position'], kind = 'scatter', xlim = (-0.5, 7), ylim = (7726, 7736), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['goodness of fit'], data['peak position'], kind = 'kde', xlim = (-0.5, 7), ylim = (7726, 7736), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')

# sns.jointplot(data['goodness of fit'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.5, 7), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['goodness of fit'], data['pre-edge noise'], kind = 'kde', xlim = (-0.5, 7), ylim = (0.006, 0.035), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("goodness of fit", "")
plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')

#

# sns.jointplot(data['peak height'], data['peak position'], kind = 'scatter', xlim = (-1, 2.2), ylim = (7726, 7736), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['peak height'], data['peak position'], kind = 'kde', xlim = (-1, 2.2), ylim = (7726, 7736), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')

# sns.jointplot(data['peak height'], data['pre-edge noise'], kind = 'scatter', xlim = (-1, 2.2), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['peak height'], data['pre-edge noise'], kind = 'kde', xlim = (-1, 2.2), ylim = (0.006, 0.035), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("peak height", "")
plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')

#

# sns.jointplot(data['peak position'], data['pre-edge noise'], kind = 'scatter', xlim = (7726, 7736), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['peak position'], data['pre-edge noise'], kind = 'kde', xlim = (7726, 7736), ylim = (0.006, 0.035), dropna = True, n_levels = 30)
plt.tight_layout()
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("peak position", "")
plt.savefig(os.path.join(save_path, 'peak position' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')
