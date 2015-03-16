#!/usr/bin/env python

"""
@file gen_image_statistic.py
@brief generate image statistic about the width/height
@author ChenglongChen
"""

import sys
import numpy as np
from skimage.io import imread
from collections import defaultdict
from matplotlib import pyplot as plt

def gen_image_statistic(task, folder_in, list_file_in, pdf_file):
    # collect argv
    list_in = np.loadtxt(list_file_in, dtype=str)
    # generate transformed image
    cnt = 0
    dict_ = defaultdict(lambda : 0)
    for file_in,label in zip(list_in[:,0], list_in[:,1]):
        image_file = folder_in + file_in
        image = imread(image_file, as_grey=True)
        image = image.copy()
        height, width = image.shape[0], image.shape[1]
        for s in [height, width]:
            dict_[s] += 1
        # report progress
        cnt += 1
        if cnt%1000 == 0:
            print "Processed %s files." % cnt
            #break
            
    x = dict_.keys()
    y = dict_.values()
    x, y = np.asarray(x), np.asarray(y)
    #y = y / np.sum(y)
    # logloss
    plt.plot(x, y)
    plt.xlabel("Length (width / height)")
    plt.ylabel("Freq")
    ls = "Max at %s" % x[np.where(y == np.max(y))[0][0]]
    plt.legend([ls], loc="best")
    plt.savefig(pdf_file)
    print( "Save pdf figure to %s" % pdf_file )
         
if __name__ == "__main__":

    # collect argv
    task = sys.argv[1]
    folder_in = sys.argv[2]
    list_file_in = sys.argv[3]
    pdf_file = sys.argv[4]
    gen_image_statistic(task, folder_in, list_file_in, pdf_file)