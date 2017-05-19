"""
author: Fang Ren (SSRL)

5/2/2017
"""


import numpy as np
from import_features import import_features
from compress_image import compress_image

def zero_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack):
    # applying mask when the value is zero, set it to nan, apply the same masks to all the
    mask = (edgejumpmap == 0)+(edgeposition_stack == 0)+(peak_stack == 0)+(goodness_of_fit == 0)+(noisemap_stack == 0)+(peak_height == -1)
    edgejumpmap[mask] = np.nan
    edgeposition_stack[mask] = np.nan
    peak_stack[mask] = np.nan
    goodness_of_fit[mask] = np.nan
    noisemap_stack[mask] = np.nan
    peak_height[mask] = np.nan

    return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack


# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = import_features()
# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = compress_image(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = zero_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
