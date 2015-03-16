#!/usr/bin/env python

"""
@file gen_submission.py
@brief add header and index to the predicted probability, as submission in Kaggle format
@author ChenglongChen
"""

import sys
import csv
import numpy as np
import pandas as pd

if __name__ == "__main__":
    sample_file = sys.argv[1]
    test_list_file = sys.argv[2]
    prediction_file = sys.argv[3]
    submission_file = sys.argv[4]

    fc = csv.reader(file(sample_file))
    header = fc.next()[1:]

    test_image_list = np.loadtxt(test_list_file, dtype=str)
    test_image_list = test_image_list[:,0]

    prediction = pd.read_csv(prediction_file, header=None)
    prediction.columns = header
    prediction.index = test_image_list
    prediction.index.name = "image"

    prediction.to_csv(submission_file)