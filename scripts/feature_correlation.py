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
from compress_image import compress_image
from odd_value_mask import odd_value_mask

#########################
# import 1st particle
feature_path1 = 'data\\particle1\\'
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = import_features(feature_path1)
edgeposition_stack1 = edgeposition_stack1
#########################

#########################
# import 2nd particle
feature_path2 = 'data\\particle2\\'
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = import_features(feature_path2)
#########################
#
#########################
# import 3rd particle
feature_path3 = 'data\\particle3\\'
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = import_features(feature_path3)
#########################

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

#########################
# flatten images
edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1 = flatten_imArray(edgejumpmap1, edgeposition_stack1, goodness_of_fit1, peak_height1, peak_stack1, noisemap_stack1)
edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2 = flatten_imArray(edgejumpmap2, edgeposition_stack2, goodness_of_fit2, peak_height2, peak_stack2, noisemap_stack2)
edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3 = flatten_imArray(edgejumpmap3, edgeposition_stack3, goodness_of_fit3, peak_height3, peak_stack3, noisemap_stack3)
#########################

#########################
# conbine three particle data
edgejumpmap = edgejumpmap1 + edgejumpmap2 + edgejumpmap3
edgeposition_stack = edgeposition_stack1 + edgeposition_stack2 + edgeposition_stack3
peak_stack = peak_stack1 + peak_stack2 + peak_stack3
goodness_of_fit = goodness_of_fit1 + goodness_of_fit2 + goodness_of_fit3
noisemap_stack = noisemap_stack1 + noisemap_stack2 + noisemap_stack3
peak_height = peak_height1 + peak_height2 + peak_height3

goodness_of_fit = np.array(goodness_of_fit)
goodness_of_fit[goodness_of_fit>0.5] = np.nan

data = pd.DataFrame({'edge jump': edgejumpmap, 'edge position':edgeposition_stack, 'peak position':peak_stack, 'goodness of fit':goodness_of_fit, 'pre-edge noise':noisemap_stack, 'peak height':peak_height})
data = data.astype(float)
#########################

# print np.count_nonzero(~np.isnan(edgejumpmap))
# print np.count_nonzero(~np.isnan(edgeposition_stack))
# print np.count_nonzero(~np.isnan(peak_stack))
# print np.count_nonzero(~np.isnan(goodness_of_fit))
# print np.count_nonzero(~np.isnan(noisemap_stack))
# print np.count_nonzero(~np.isnan(peak_height))


print 'visualization'
# visualization
save_path = 'C:\\Research_FangRen\\Python codes\\feature_extraction_TXM\\report\\variable_correlation'
if not os.path.exists(save_path):
    os.mkdir(save_path)
#
# # print goodness_of_fit
# sns.kdeplot(np.nan_to_num(np.array(goodness_of_fit)))
# plt.xlim(0, 120)
# plt.savefig(os.path.join(save_path, 'goodness_of_fit'))

names = list(data)
corr = data.corr()
# print corr

plt.matshow(abs(corr).T, cmap = 'viridis')
print corr.columns
plt.xticks(range(len(corr.columns)), ['H$_{edge}$','E$_{edge}$','$\chi^2$','H$_{peak}$','E$_{peak}$','$\sigma$'])
plt.yticks(range(len(corr.columns)), ['H$_{edge}$','E$_{edge}$','$\chi^2$','H$_{peak}$','E$_{peak}$','$\sigma$'])
plt.colorbar()
plt.savefig(os.path.join(save_path, 'correlation'), dpi = 600)
plt.close('all')

sns.set(font_scale=1.7, style = 'white')

#
# #sns.jointplot(data['edge jump'], data['edge position'], kind = 'scatter', xlim = (-0.01, 0.7), ylim = (7714, 7727), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge jump'], data['edge position'], kind = 'kde', xlim = (-0.01, 0.7), ylim = (7714, 7727), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "E$_{edge}$")
# plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'edge position'), dpi = 600)
# plt.close('all')
#
# # sns.jointplot(data['edge jump'], data['peak height'], kind = 'scatter', xlim = (-0.01, 0.7), ylim = (-0.3, 1.5), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge jump'], data['goodness of fit'], kind = 'kde', xlim = (-0.01, 0.7), ylim = (-0.01, 0.1), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "$\chi^2$")
# plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'goodness of fit'), dpi = 600)
# plt.close('all')
#
# # sns.jointplot(data['edge jump'], data['peak height'], kind = 'scatter', xlim = (-0.01, 0.7), ylim = (-0.3, 1.5), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge jump'], data['peak height'], kind = 'kde', xlim = (-0.01, 0.7), ylim = (-0.3, 1.5), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "H$_{peak}$")
# plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak height'), dpi = 600)
# plt.close('all')

# sns.jointplot(data['edge jump'], data['peak position'], kind = 'scatter', xlim = (-0.01, 0.7), ylim = (7727, 7732  ), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge jump'], data['peak position'], kind = 'kde', xlim = (-0.01, 0.7), ylim = (7727, 7732 ), dropna = True, n_levels = 10)
plt.tight_layout()
g.ax_joint.set_xticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "E$_{peak}$")
plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')
#
# # sns.jointplot(data['edge jump'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.01, 0.7), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge jump'], data['pre-edge noise'], kind = 'kde', xlim = (-0.01, 0.7), ylim = (0.006, 0.035), dropna = True, n_levels = 10)
# plt.tight_layout()
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("H$_{edge}$", "$\sigma$")
# plt.savefig(os.path.join(save_path, 'edge jump' + ' Vs ' + 'pre-edge noise'), dpi = 600)
# plt.close('all')
#
#
#
# # sns.jointplot(data['edge position'], data['peak height'], kind = 'scatter', xlim = (7714, 7727), ylim = (-0.3, 1.5), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge position'], data['goodness of fit'], kind = 'kde', xlim = (7714, 7727), ylim = (-0.01, 0.1), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "")
# plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'goodness of fit'), dpi = 600)
# plt.close('all')
#
#
# # sns.jointplot(data['edge position'], data['peak height'], kind = 'scatter', xlim = (7714, 7727), ylim = (-0.3, 1.5), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge position'], data['peak height'], kind = 'kde', xlim = (7714, 7727), ylim = (-0.3, 1.5), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "")
# plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak height'), dpi = 600)
# plt.close('all')

# sns.jointplot(data['edge position'], data['peak position'], kind = 'scatter', xlim = (7714, 7727), ylim = (7727, 7732  ), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['edge position'], data['peak position'], kind = 'kde', xlim = (7714, 7727), ylim = (7727, 7732  ), dropna = True, n_levels = 10)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')



# # sns.jointplot(data['edge position'], data['pre-edge noise'], kind = 'scatter', xlim = (7714, 7727), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['edge position'], data['pre-edge noise'], kind = 'kde', xlim = (7714, 7727), ylim = (0.006, 0.035), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("E$_{edge}$", "")
# plt.savefig(os.path.join(save_path, 'edge position' + ' Vs ' + 'pre-edge noise'), dpi = 600)
# plt.close('all')
#
#
#
# # sns.jointplot(data['goodness of fit'], data['peak height'], kind = 'scatter', xlim = (-0.01, 0.2), ylim = (-0.3, 1.5), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['goodness of fit'], data['peak height'], kind = 'kde', xlim = (-0.01, 0.1), ylim = (-0.3, 1.5), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_xticks([])
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("", "")
# plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'peak height'), dpi = 600)
# plt.close('all')
# #

# sns.jointplot(data['goodness of fit'], data['peak position'], kind = 'scatter', xlim = (-0.01, 0.2), ylim = (7727, 7732  ), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['goodness of fit'], data['peak position'], kind = 'kde', xlim = (-0.01, 0.1), ylim = (7727, 7732  ), dropna = True, n_levels = 10)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')
#
# # sns.jointplot(data['goodness of fit'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.01, 0.2), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['goodness of fit'], data['pre-edge noise'], kind = 'kde', xlim = (-0.01, 0.1), ylim = (0.006, 0.035), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("$\chi^2$", "")
# plt.savefig(os.path.join(save_path, 'goodness of fit' + ' Vs ' + 'pre-edge noise'), dpi = 600)
# plt.close('all')
#
# #

# sns.jointplot(data['peak height'], data['peak position'], kind = 'scatter', xlim = (-0.3, 1.5), ylim = (7727, 7732  ), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['peak height'], data['peak position'], kind = 'kde', xlim = (-0.3, 1.5), ylim = (7727, 7732  ), dropna = True, n_levels = 10)
plt.tight_layout()
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("", "")
plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'peak position'), dpi = 600)
plt.close('all')

# # sns.jointplot(data['peak height'], data['pre-edge noise'], kind = 'scatter', xlim = (-0.3, 1.5), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
# g = sns.jointplot(data['peak height'], data['pre-edge noise'], kind = 'kde', xlim = (-0.3, 1.5), ylim = (0.006, 0.035), dropna = True, n_levels = 10)
# plt.tight_layout()
# g.ax_joint.set_yticks([])
# # g.ax_marg_x.set_axis_off()
# # g.ax_marg_y.set_axis_off()
# g.set_axis_labels("H$_{peak}$", "")
# plt.savefig(os.path.join(save_path, 'peak height' + ' Vs ' + 'pre-edge noise'), dpi = 600)
# plt.close('all')
#
# #

# sns.jointplot(data['peak position'], data['pre-edge noise'], kind = 'scatter', xlim = (7727, 7732  ), ylim = (0.006, 0.0425), joint_kws={"s": 5}, dropna = True, marginal_kws={"bins":100})
g = sns.jointplot(data['peak position'], data['pre-edge noise'], kind = 'kde', xlim = (7727, 7732  ), ylim = (0.006, 0.035), dropna = True, n_levels = 10)
plt.tight_layout()
g.ax_joint.set_yticks([])
# g.ax_marg_x.set_axis_off()
# g.ax_marg_y.set_axis_off()
g.set_axis_labels("E$_{peak}$", "")
plt.savefig(os.path.join(save_path, 'peak position' + ' Vs ' + 'pre-edge noise'), dpi = 600)
plt.close('all')
