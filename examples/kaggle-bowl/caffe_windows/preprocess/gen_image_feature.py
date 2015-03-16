#!/usr/bin/env python

"""
@file gen_image_feature.py
@brief generate handcrafted image features
@author ChenglongChen
"""

import sys
import csv
import numpy as np
from string import atoi
from features import extractFeatV1, extractFeatV2, extractFeatV3


def gen_image_feature(version, task, folder_in, list_file_in, feat_file_out):
    # collect argv
    list_in = np.loadtxt(list_file_in, dtype=str)
    feat_out_file_ = csv.writer(open(feat_file_out, "w"), delimiter='\t', lineterminator='\n')

    # generate transformed image
    cnt = 0
    for file_in,label in zip(list_in[:,0], list_in[:,1]):        
        if version == 1:
            # version 1
            feat = extractFeatV1(folder_in + file_in)
        elif version == 2:
            # version 2
            feat = extractFeatV2(folder_in + file_in)
        elif version == 3:
            # version 3
            feat = extractFeatV3(folder_in + file_in)
        feat_out_file_.writerow( [file_in] + feat.tolist() )
        # report progress
        cnt += 1
        if cnt%1000 == 0:
            print "Processed %s files." % cnt
         
if __name__ == "__main__":

    # collect argv
    version = atoi(sys.argv[1])
    task = sys.argv[2]
    folder_in = sys.argv[3]
    list_file_in = sys.argv[4]
    feat_file_out = sys.argv[5]
    gen_image_feature(version, task, folder_in, list_file_in, feat_file_out)