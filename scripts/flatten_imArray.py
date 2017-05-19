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
from import_features import import_features
from zero_mask import zero_mask
from outlier_mask import outlier_mask

def flatten_imArray(edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack):
    s1 = edgejumpmap.shape[0]
    s2 = edgejumpmap.shape[1]

    edgejumpmap = list(edgejumpmap.reshape(s1*s2, 1)[:,0])
    edgeposition_stack = list(edgeposition_stack.reshape(s1*s2, 1)[:,0])
    peak_stack = list(peak_stack.reshape(s1*s2, 1)[:,0])
    goodness_of_fit = list(goodness_of_fit.reshape(s1*s2, 1)[:,0])
    noisemap_stack = list(noisemap_stack.reshape(s1*s2, 1)[:,0])
    peak_height = list(peak_height.reshape(s1*s2, 1)[:,0])

    return edgejumpmap, edgeposition_stack, goodness_of_fit, peak_height, peak_stack, noisemap_stack

