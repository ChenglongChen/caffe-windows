#!/usr/bin/env python

"""
@file plot_caffe_training_logloss.py
@brief plot caffe training logloss (coarse and fine label)
@author ChenglongChen
"""

import re
import sys
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import figaspect

def fetchPattern(pat, log_file):
    m = re.search(pat, open(log_file, "r").read())
    v = int(m.group(1))
    return v

def main():

    # collect argvs
    log_file = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_file = sys.argv[2]
    else:
        pdf_file = sys.argv[1][:-4]+".pdf"

    # fetch training and validation iteration
    cmd = "cat %s | grep 'solver.cpp:231] Iteration ' | awk '{print $6}' | awk -F',' '{print $1}' > train_iteration.tmp" % log_file
    subprocess.call(cmd, shell=True)
    cmd = "cat %s | grep 'solver.cpp:287] Iteration ' | awk '{print $6}' | awk -F',' '{print $1}' > valid_iteration.tmp" % log_file
    subprocess.call(cmd, shell=True)
    
    # fetch training and validation logloss
    cmd = "cat %s | grep 'Train net output #3: loss_fine = ' | awk '{print $11}' > train_logloss.tmp" % log_file
    subprocess.call(cmd, shell=True)
    cmd = "cat %s | grep 'Test net output #3: loss_fine =' | awk '{print $11}' > valid_logloss.tmp" % log_file
    subprocess.call(cmd, shell=True)

    # fetch training and validation accuracy
    cmd = "cat %s | grep 'Train net output #1: accuracy_fine = ' | awk '{print $11}' > train_accuracy.tmp" % log_file
    subprocess.call(cmd, shell=True)
    cmd = "cat %s | grep 'Test net output #1: accuracy_fine =' | awk '{print $11}' > valid_accuracy.tmp" % log_file
    subprocess.call(cmd, shell=True)

    train_logloss = np.loadtxt("train_logloss.tmp")    
    valid_logloss = np.loadtxt("valid_logloss.tmp")
    train_accuracy = 100*np.loadtxt("train_accuracy.tmp")
    valid_accuracy = 100*np.loadtxt("valid_accuracy.tmp")

    train_iteration = np.loadtxt("train_iteration.tmp",dtype=int)
    valid_iteration = np.loadtxt("valid_iteration.tmp",dtype=int)
    valid_logloss_min = np.min(valid_logloss)
    valid_accuracy_min = np.min(valid_accuracy)
    best_ind = np.where(valid_logloss == valid_logloss_min)[0][0]
    best_iter = valid_iteration[best_ind]


    # plot training and validation logloss
    w, h = figaspect(0.5)
    plt.figure(figsize=(w,h))
    f, axarr = plt.subplots(2, sharex=True)
    # logloss
    axarr[0].plot(train_iteration, train_logloss)
    axarr[0].plot(valid_iteration, valid_logloss)
    axarr[0].plot(train_iteration, valid_logloss_min * np.ones((len(train_iteration))))
    axarr[0].set_title("LogLoss vs Iteration")
    axarr[0].set_ylabel("LogLoss")
    axarr[0].legend(["Train", "Valid (Min = %s at Iter. = %s)" % (np.round(valid_logloss_min,5), best_iter)], loc="upper right")
    # accuracy
    axarr[1].plot(train_iteration, train_accuracy)
    axarr[1].plot(valid_iteration, valid_accuracy)
    axarr[1].plot(train_iteration, valid_accuracy[best_ind] * np.ones((len(train_iteration))))
    axarr[1].set_title("Accuracy vs Iteration")
    axarr[1].set_xlabel("Iteration")
    axarr[1].set_ylabel("Accuracy [%]")
    axarr[1].legend(["Train", "Valid (Acc = %s%% at Iter. = %s)" % (np.round(valid_accuracy[best_ind],2), best_iter)], loc="lower right")
    #fmt = '%.0f%%'
    #yticks = mtick.FormatStrFormatter(fmt)
    #axarr[1].yaxis.set_major_formatter(yticks)
    #plt.tight_layout()
    plt.savefig(pdf_file)
    print( "Save pdf figure to %s" % pdf_file )

    # cleanup
    cmd = "rm -rf train_iteration.tmp"
    subprocess.call(cmd, shell=True)
    cmd = "rm -rf valid_iteration.tmp"
    subprocess.call(cmd, shell=True)
    cmd = "rm -rf train_logloss.tmp"
    subprocess.call(cmd, shell=True)
    cmd = "rm -rf valid_logloss.tmp"
    subprocess.call(cmd, shell=True)
    cmd = "rm -rf train_accuracy.tmp"
    subprocess.call(cmd, shell=True)
    cmd = "rm -rf valid_accuracy.tmp"
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()
