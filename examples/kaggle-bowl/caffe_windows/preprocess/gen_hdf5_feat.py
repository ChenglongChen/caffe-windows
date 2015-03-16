#!/usr/bin/env python

"""
@file gen_hdf5_feat.py
@brief dump handcrafted features into hdf5 format
@author ChenglongChen
"""

import sys
import csv
import h5py
import numpy as np
import pandas as pd
from string import atoi
import warnings
warnings.filterwarnings("ignore")


def genLabelMapper(sample_submission_file):
    """This function generates mapper for mapping fine-grained label to
    coarse-grained label using the sample submission and/or the tree"""
 
    # read in header
    header = csv.reader( file(sample_submission_file) ).next()
    header = header[1:]

    # make class map
    label_mapper = {}
    coarse_labels = [ header[0].split("_")[0] ]
    cnt = 0
    for i,h in enumerate(header):
        h2 = h.split("_")[0]
        if not h2 in coarse_labels:
            cnt += 1
            coarse_labels.append(h2)
        label_mapper[h] = cnt

    return label_mapper


def applyLabelMap(label_string, label_mapper):
    """This function applies mapping from fine-grained label to
    coarse-grained label using the given label mapper"""

    label = np.zeros( (label_string.shape[0],1), dtype="float32" )
    for i,f in enumerate(label_string):
        label[i] = label_mapper[ f.split("/")[0] ]

    return label


def main():
    """Main function"""

    # collect argvs
    sample_submission_file = sys.argv[1]
    feature_file = sys.argv[2]
    hdf5_out = sys.argv[3]
    chunksize = atoi(sys.argv[4])
    task = sys.argv[5]

    # generate label mapper
    label_mapper = genLabelMapper(sample_submission_file)

    # check the size of the feature matrix: num_sample x num_feat
    num_sample = len( open(feature_file, "rb").readlines() )
    num_feat = len( open(feature_file, "rb").readline().split('\t') ) - 1
    print "Convert HDF5] #Sample: %s" % num_sample
    print "Convert HDF5] #Feature: %s" % num_feat

    # read in feature matrix in chunk
    reader = pd.read_table( feature_file, sep='\t', chunksize=chunksize, header=None )
    cnt = 0
    h5py_files = []
    for i,chunk in enumerate( reader ):        
        val = chunk.values
        num = val.shape[0]
        feat = np.asarray( val[:, 1:, np.newaxis, np.newaxis], dtype='float32' )
        if task == "test":
            label = np.zeros((num,1), dtype='float32' )
        else:
            label = np.asarray( applyLabelMap( val[:, 0], label_mapper ), dtype='float32' )
        # create HDF5 dataset
        file = hdf5_out + "_chunk" + str(i)
        h5py_files.append( file )
        with h5py.File(file, 'w') as f:
            f['data'] = feat
            f['label'] = label
        cnt += num        
        print "Convert HDF5] Processed %s samples." % cnt

    # create HDF5 source file
    with open(hdf5_out + '.list', 'w') as f:
        for file in h5py_files:
            f.write(file + '\n')


if __name__ == "__main__":
   main()