#!/usr/bin/env python

"""
@file plot_caffe_ensemble_logloss.py
@brief plot caffe ensemble logloss (single vs ensemble)
@author ChenglongChen
"""

import sys
import numpy as np
from matplotlib import pyplot as plt

def main():

    # collect argvs
    log_file_single = sys.argv[1]
    log_file_ensemble = sys.argv[2]

    if len(sys.argv) > 3:
        pdf_file = sys.argv[3]
    else:
        pdf_file = sys.argv[1].split("caffemodel")[0] + "caffemodel_valid_logloss.pdf"

    lines_single = open(log_file_single, "r").readlines()
    logloss_single = [float(l[:-4]) for l in lines_single[1::2]]
    logloss_single_mean = np.mean(logloss_single)
    logloss_single_std = np.std(logloss_single)

    lines_ensemble = open(log_file_ensemble, "r").readlines()
    logloss_ensemble = [float(l[:-4]) for l in lines_ensemble[1::2]]
    logloss_ensemble_min = np.min(logloss_ensemble)


    # logloss
    plt.plot(range(len(logloss_single)), logloss_single)
    plt.plot(range(len(logloss_ensemble)), logloss_ensemble)
    plt.plot(range(len(logloss_ensemble)), logloss_ensemble_min * np.ones((len(logloss_ensemble))))
    plt.title("LogLoss vs Number of predictions")
    plt.xlabel("Number of predictions")
    plt.ylabel("LogLoss")
    ls = "Single (Mean = %s, Std = %s)" % (np.round(logloss_single_mean,5), np.round(logloss_single_std,5))
    le = "Ensemble (Min = %s)" % np.round(logloss_ensemble_min,5)
    plt.legend([ls, le], loc="best")

    plt.savefig(pdf_file)
    print( "Save pdf figure to %s" % pdf_file )

if __name__ == "__main__":
    main()
