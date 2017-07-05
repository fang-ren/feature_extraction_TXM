"""
author: Fang Ren (SSRL)

5/2/2017
"""


import numpy as np
from import_features import import_features
from compress_image import compress_image

def odd_value_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack):
    # applying mask when the value is zero, set it to nan, apply the same masks to all the
    mask = (peak_stack > 7734) + (peak_stack < 7725) + (edgeposition_stack > 7727) + (edgeposition_stack < 7714)
    # peak_stack[peak_stack > 7734] = 7729.5
    # peak_stack[peak_stack < 7725] = 7729.5
    # edgeposition_stack[edgeposition_stack > 7727] = 7722
    # edgeposition_stack[edgeposition_stack < 7714] = 7722

    edgejumpmap[mask] = np.nan
    edgeposition_stack[mask] = np.nan
    peak_stack[mask] = np.nan
    goodness_of_fit[mask] = np.nan
    noisemap_stack[mask] = np.nan
    peak_height[mask] = np.nan

    return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack

if __name__ == '__main__':
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = import_features()
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = compress_image(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
    edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = odd_value_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
