#!/usr/bin/env python

"""
@file compute_accuracy.py
@brief compute accuracy
@author ChenglongChen
"""

import sys
import numpy as np
import pandas as pd

def computeAccuracy(prob, label):
    pred = np.argmax(prob, axis=1)
    accuracy = np.mean(pred == label)
    return accuracy

def main():
    list_file = sys.argv[1]
    prob_file = sys.argv[2]
    
    list_in = np.loadtxt(list_file, dtype=str)
    true_label = np.asarray(list_in[:,1], dtype="int")
    
    prob = pd.read_csv(prob_file, index_col=0).values
    accuracy = computeAccuracy(prob, true_label)
    print( accuracy )
    
if __name__ == "__main__":
    main()