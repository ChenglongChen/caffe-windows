#!/usr/bin/env python

"""
@file fuse_submission.py
@brief fuse two submissions
@author ChenglongChen
"""

import sys
import pandas as pd

def fuse_submission(fin1, weight1, fin2, weight2, fout):
    p1 = pd.read_csv(fin1, index_col=0)
    p2 = pd.read_csv(fin2, index_col=0)
    p = (weight1 * p1 + weight2 * p2) / (weight1 + weight2)
    p = pd.DataFrame(p, columns=p1.columns, index=p1.index)
    p.index.name = p1.index.name
    p.to_csv(fout)
    
    
def main():
    fin1 = sys.argv[1]
    weight1 = float(sys.argv[2])
    fin2 = sys.argv[3]
    weight2 = float(sys.argv[4])
    fout = sys.argv[5]
    fuse_submission(fin1, weight1, fin2, weight2, fout)

if __name__ == "__main__":
    main()