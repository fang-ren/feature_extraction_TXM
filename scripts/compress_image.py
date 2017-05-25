"""
author: Fang Ren (SSRL)

5/2/2017
"""

import numpy as np
from import_features import import_features

def compress_image(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack):
        # compress the image for speeding up
        keep = np.arange(0, 400, 5)
        edgejumpmap = edgejumpmap[keep][:, keep]
        edgeposition_stack = edgeposition_stack[keep][:, keep]
        peak_stack = peak_stack[keep][:, keep]
        goodness_of_fit = goodness_of_fit[keep][:, keep]
        noisemap_stack = noisemap_stack[keep][:, keep]
        peak_height = peak_height[keep][:, keep]

        return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack

# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = import_features('..\\data\\particle1\\')
# edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack = compress_image(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack)
# print edgejumpmap.shape
