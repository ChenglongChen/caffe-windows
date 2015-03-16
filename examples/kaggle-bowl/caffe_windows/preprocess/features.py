#!/usr/bin/env python

"""
@file features.py
@brief provide functions to compute image features
@author ChenglongChen
"""

import numpy as np
from numpy import pi
from skimage.io import imread
from skimage import measure
from skimage import morphology
from skimage.feature import greycomatrix, greycoprops
import warnings
warnings.filterwarnings("ignore")
import mahotas as mh
from mahotas.features import surf
from scipy.stats.mstats import mquantiles, kurtosis, skew

def tryDivide(x, y):
    if y == 0:
        return 0.0
    else:
        return x / y

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
    regionmaxprop = None
    for regionprop in props:
        # check to see if the region is at least 50% nonzero
        if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    return regionmaxprop

def estimateRotationAngle(image_file):
    image = imread(image_file, as_grey=True)
    image = image.copy()
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    angle = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        angle = 0.0 if maxregion is None else maxregion.orientation*180.0/pi
    return -angle


def extractFeatV1(image_file):
    image = imread(image_file, as_grey=True)
    image = image.copy()
        
    # Create the thresholded image to eliminate some of the background
    imagethr = np.where(image > np.mean(image),0.,1.0)

    #Dilate the image
    imdilated = morphology.dilation(imagethr, np.ones((4,4)))

    # Create the label list
    label_list = measure.label(imdilated)
    label_list = imagethr*label_list
    label_list = label_list.astype(int)
    
    region_list = measure.regionprops(label_list, image)
    maxregion = getLargestRegion(region_list, label_list, imagethr)
    
    # guard against cases where the segmentation fails by providing zeros
    #angle = 0.0
    minor_axis = 0.0
    major_axis = 0.0
    area = 0.0
    convex_area = 0.0
    perimeter = 0.0
    equivalent_diameter = 0.0
    solidity = 0.0
    eccentricity = 0.0
    extent = 0.0
    mean_intensity = 0.0
    weighted_moments_normalized = [0.0] * 13
    exclude = [0, 1, 4] # this indices contain NaN
    if not maxregion is None:
        #angle = maxregion.orientation*180.0/pi
        minor_axis = maxregion.minor_axis_length
        major_axis = maxregion.major_axis_length
        area = maxregion.area
        convex_area = maxregion.convex_area
        extent = maxregion.extent
        perimeter = maxregion.perimeter
        equivalent_diameter = maxregion.perimeter
        solidity = maxregion.solidity
        eccentricity = maxregion.eccentricity
        mean_intensity = maxregion.mean_intensity
        #print maxregion.weighted_moments_normalized.shape
        tmp = maxregion.weighted_moments_normalized.reshape((16))
        tmp = np.nan_to_num(tmp)
        weighted_moments_normalized = [tmp[i] for i in range(16) if i not in exclude]

    axis_ratio = tryDivide(minor_axis, major_axis)

    region_feat = [
        axis_ratio,
        minor_axis,
        major_axis,
        area,
        convex_area,
        extent,
        perimeter,
        equivalent_diameter,
        solidity,
        eccentricity,
        mean_intensity,
    ]
    region_feat += weighted_moments_normalized

    # concat all the features
    feat = region_feat
    feat = np.asarray(feat, dtype="float32")
    
    return feat
    
def gini(x,f=""):
    """
    http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html
    """
    # requires all values in x to be zero or positive numbers,
    # otherwise results are undefined
    x = x.flatten()
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
    if s == 0 or n == 0:
        print "GINI debug",f
        return 1.0
    else:
        return 1.0 - (2.0 * (r*x).sum() + s)/(n*s)
        
def extractFeatV2(image_file):
    region_feat = extractFeatV1(image_file)
    
    image = imread(image_file, as_grey=True)
    image = image.copy()

    # global features
    idx = np.nonzero(255-image)
    nonzero = image[idx]
    global_feat = [ 
        np.mean(nonzero),
        np.std(nonzero),
        kurtosis(nonzero),
        skew(nonzero),
        gini(nonzero,image_file),
    ]
    global_feat = np.asarray(global_feat, dtype='float32' )

    # concat all the features
    image2 = mh.imread(image_file, as_grey=True)
    haralick = mh.features.haralick(image2, ignore_zeros=False, preserve_haralick_bug=False, compute_14th_feature=False)
    lbp = mh.features.lbp(image2, radius=20, points=7, ignore_zeros=False)
    pftas = mh.features.pftas(image2)
    zernike_moments = mh.features.zernike_moments(image2, radius=20, degree=8)
    #surf_feat = surf.surf(image2)
    haralick = np.reshape(haralick,(np.prod(haralick.shape)))
    #surf_feat = np.reshape(surf_feat,(np.prod(surf_feat.shape)))
    
    #mh_feat = np.hstack((haralick, lbp, pftas, zernike_moments, surf_feat))
    mh_feat = np.hstack((haralick, lbp, pftas, zernike_moments))

    feat = np.hstack((global_feat, region_feat, mh_feat))
    
    return feat
    
def extractFeatV3(image_file):
    feat = extractFeatV1(image_file)
    
    image = imread(image_file, as_grey=True)
    image = image.copy()
    glcm = greycomatrix(image,
                        distances=[1, 2, 4, 8],
                        angles=[0, pi/4, pi/2, 3*pi/4],
                        levels=256,
                        normed=True,
                        symmetric=True)
    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']
    for p in props:
        res = greycoprops(glcm, p)
        feat = np.hstack((feat, res.flatten()))
        
    return feat
