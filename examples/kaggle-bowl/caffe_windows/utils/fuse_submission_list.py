#!/usr/bin/env python

"""
@file fuse_submission_list.py
@brief fuse a list of submissions
@author ChenglongChen
"""

import sys
import numpy as np
import pandas as pd

def fuse_submission_list(fout, fins, weights):
    p0 = pd.read_csv(fins[-1], index_col=0)
    numTest = p0.shape[0]
    numLabel = p0.shape[1]
    p_ens = np.zeros((numTest, numLabel), dtype="float")
    for f,w in zip(fins, weights):
        this_p = pd.read_csv(f, index_col=0).values[-numTest:]
        p_ens += w * this_p
    p_ens /= np.sum(weights)
    p = pd.DataFrame(p_ens, columns=p0.columns, index=p0.index)
    p.index.name = p0.index.name
    p.to_csv(fout)
    
    
def main():
    fout = sys.argv[1]
    fins = sys.argv[2::2]
    weights = np.asarray(sys.argv[3::2], dtype="float")
    print fins
    print weights
    fuse_submission_list(fout, fins, weights)

if __name__ == "__main__":
    main()