"""
author: Fang Ren (SSRL)

5/2/207
"""

import numpy as np

def outlier_mask(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack):
    # remove outliers using 3 sigma, replace it with nan
    mask = (abs(edgeposition_stack - np.nanmean(edgeposition_stack)) > 3 * np.nanstd(edgeposition_stack)) + \
            (abs(peak_stack - np.nanmean(peak_stack)) > 3 * np.nanstd(peak_stack)) + \
            (abs(goodness_of_fit - np.nanmean(goodness_of_fit)) > 3 * np.nanstd(goodness_of_fit))
    
    edgejumpmap[mask] = np.nan
    edgeposition_stack[mask] = np.nan
    peak_stack[mask] = np.nan
    goodness_of_fit[mask] = np.nan
    noisemap_stack[mask] = np.nan
    peak_height[mask] = np.nan

    return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack