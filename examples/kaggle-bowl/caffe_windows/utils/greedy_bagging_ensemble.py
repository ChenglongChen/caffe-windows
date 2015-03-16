#!/usr/bin/env python

"""
@file greedy_bagging_ensemble.py
@brief perform greedy bagging ensemble based on validation logloss (mind the chance of overfitting!)
@author ChenglongChen
"""

import sys
import numpy as np
import pandas as pd
from copy import copy
sys.path.append("../utils")
from compute_logloss import computeLogloss
from compute_accuracy import computeAccuracy
from collections import defaultdict

def greedyWeight(p1, weight1, p2, weight2_range, true_label):
    best_weight2 = 1.0
    best_logloss = 10
    for weight2 in weight2_range:
        p = (weight1 * p1 + weight2 * p2) / (weight1 + weight2)
        logloss = computeLogloss(p, true_label)
        #print( "w1: %s, w2: %s, logloss: %s" % (weight1, weight2, logloss) )
        if logloss < best_logloss:
            best_logloss, best_weight2 = logloss, weight2
    return best_weight2
    
def greedyEnsemble(list_file, fout, mode, task, fins):
    
    list_in = np.loadtxt(list_file, dtype=str)
    true_label = np.asarray(list_in[:,1], dtype="int") 
    numValid = true_label.shape[0]
    numTest = 130400
    
    p0_valid = pd.read_csv(fins[-1] + "_valid.csv", index_col=0)
    
    numLabel = p0_valid.shape[1]
    p_ens_valid = np.zeros((numValid, numLabel), dtype="float")
    if task == "test":
        p0_test = pd.read_csv(fins[-1] + "_test.csv", index_col=0)
        p_ens_test = np.zeros((numTest, numLabel), dtype="float")
    
    best_dict = defaultdict(lambda : 0)
    fins_tmp = copy(fins)
    
    if mode == "average":
        # naive average
        for f in fins:
            this_p_valid = pd.read_csv(f + "_valid.csv", index_col=0).values[-numValid:]
            p_ens_valid += this_p_valid
            if task == "test":
                this_p_test = pd.read_csv(f + "_test.csv", index_col=0).values[-numTest:]
                p_ens_test += this_p_test
            
        p_ens_valid /= len(fins)
        if task == "test":
            p_ens_test /= len(fins)
        
        best_logloss = computeLogloss(p_ens_valid, true_label)
        print( "\nAverage result:" )
        print( "logloss: %s" % (best_logloss) )
        
    elif mode == "greedy":
        # greedy ensemble    
        best_logloss = 10
        best_accuracy = 0
        best_fin = None
        first = True
        while True:
            for f in fins_tmp:
                this_p_valid = pd.read_csv(f + "_valid.csv", index_col=0).values[-numValid:]
                if first:
                    w_ens, this_w = 0.0, 1.0
                else:
                    w_ens = 1.0
                    this_w = greedyWeight(p_ens_valid, w_ens, this_p_valid, np.arange(0.1, 2, 0.01), true_label)  
                # all the current prediction to the ensemble
                tmp = (w_ens * p_ens_valid + this_w * this_p_valid) / (w_ens + this_w)
                logloss = computeLogloss(tmp, true_label)
                accuracy = computeAccuracy(tmp, true_label)
                if logloss < best_logloss:
                    best_logloss, best_accuracy, best_fin, best_fin_w = logloss, accuracy, f, this_w
            if best_fin == None:
                break

            print best_fin
            print best_fin_w
            print best_logloss
            print best_accuracy
            best_dict[best_fin] += best_fin_w
            # valid
            this_p_valid = pd.read_csv(best_fin + "_valid.csv", index_col=0).values[-numValid:]
            p_ens_valid = (w_ens * p_ens_valid + best_fin_w * this_p_valid) / (w_ens + best_fin_w)
            # test
            if task == "test":                
                this_p_test = pd.read_csv(best_fin + "_test.csv", index_col=0).values[-numTest:]            
                p_ens_test = (w_ens * p_ens_test + best_fin_w * this_p_test) / (w_ens + best_fin_w)
            #fins_tmp.remove(best_fin)
            best_fin = None
            first = False
            
        # report the best weights and the corresponding logloss found
        print( "\nGreedy ensemble result:" )
        print( "logloss: %s" % (best_logloss) )
        print( "accuracy: %s" % (best_accuracy) )
        for i,f in enumerate(fins, start=1):
            if f not in best_dict:
                print( "        w%s=%s" % (i, 0) )
            else:
                print( "        w%s=%s" % (i, best_dict[f]) )
                
    # the final ensemble
    p_out_valid = pd.DataFrame(p_ens_valid, columns=p0_valid.columns, index=p0_valid.index)
    p_out_valid.index.name = p0_valid.index.name
    p_out_valid.to_csv( fout + "_[merge_nll" + str(np.round(best_logloss, 8)) + "]_valid.csv" )
    # test
    if task == "test": 
        p_out_test = pd.DataFrame(p_ens_test, columns=p0_test.columns, index=p0_test.index)
        p_out_test.index.name = p0_test.index.name
        p_out_test.to_csv( fout + "_[merge_nll" + str(np.round(best_logloss, 8)) + "]_test.csv" )
    
def main():
    list_file = sys.argv[1]
    fout = sys.argv[2]
    mode = sys.argv[3]
    task = sys.argv[4]
    fins = sys.argv[5:]
    greedyEnsemble(list_file, fout, mode, task, fins)

if __name__ == "__main__":
    main()