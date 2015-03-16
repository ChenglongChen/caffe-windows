#!/usr/bin/env python

"""
@file compute_logloss.py
@brief compute logloss
@author ChenglongChen
"""

import sys
import numpy as np
import pandas as pd

def softmax(score):
    num = score.shape[0]
    maxes = np.amax(score, axis=1).reshape((num, 1))
    e = np.exp(score - maxes)
    prob = e / np.sum(e, axis=1).reshape((num, 1))
    return prob

def computeLogloss(prob, label, eps=1e-15):
    # clip
    prob = np.clip(prob, eps, 1 - eps)
    # normalization
    prob /= prob.sum(axis=1)[:,np.newaxis]
    p = prob[np.arange(len(label)),label.astype(int)]
    loss = -np.mean(np.log(p))
    return loss

def main():
    list_file = sys.argv[1]
    prob_file = sys.argv[2]
    
    list_in = np.loadtxt(list_file, dtype=str)
    true_label = np.asarray(list_in[:,1], dtype="int")
    
    prob = pd.read_csv(prob_file, index_col=0).values
    if len(sys.argv) == 4 and sys.argv[3] == "raw":
        prob = softmax(prob)
    #prob = 1e-5
    #prob = sp.maximum(sp.minimum(prob, 1.0-eps), eps)
    logloss = computeLogloss(prob, true_label)
    print( logloss )
    
if __name__ == "__main__":
    main()