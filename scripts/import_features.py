"""
author: Fang Ren (SSRL)

5/2/2017
"""

from scipy.io import loadmat


def import_features(feature_path = '..\\data\\particle1\\'):
    edgejumpmap = loadmat(feature_path+ 'edgejumpmap.mat')['edgejumpmap']
    edgeposition_stack = loadmat(feature_path+ 'edgeposition_stack.mat')['edgeposition_stack']
    peak_stack = loadmat(feature_path+ 'peak_stack.mat')['peak_stack']
    goodness_of_fit = loadmat(feature_path+ 'R_squares_stack.mat')['R_squares_stack']
    noisemap_stack = loadmat(feature_path+ 'noisemap_stack.mat')['noisemap_stack']
    peak_height = loadmat(feature_path+ 'peakheight_stack.mat')['peakheight_stack']

    return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack

# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = import_features()
# print edgejumpmap.shape, edgeposition_stack.shape, goodness_of_fit.shape, peak_height.shape, peak_stack.shape, noisemap_stack.shape