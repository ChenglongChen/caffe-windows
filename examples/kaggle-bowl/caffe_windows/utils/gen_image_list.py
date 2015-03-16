#!/usr/bin/env python

"""
@file gen_image_list.py
@brief generate image list
@author ChenglongChen
"""

import os
import sys
import csv

def main():
    
    # collect argvs
    task = sys.argv[1]
    assert task in ["train", "test"]
    sample_submission_file = sys.argv[2]
    image_folder = sys.argv[3]
    file_out = sys.argv[4]

    # read in header
    header = csv.reader( file(sample_submission_file) ).next()
    header = header[1:]

    # make image list
    img_lst = []
    cnt = 0
    if task == "train":
        for i in xrange(len(header)):
            lst = os.listdir(image_folder + header[i])
            for img in lst:
                img_lst.append((header[i] + '/' + img, i))
                cnt += 1
    else:
        lst = os.listdir(image_folder)
        for img in lst:
            img_lst.append((img, 0))
            cnt += 1

    # write
    with open(file_out, "w") as f:
        writer_ = csv.writer(f, delimiter='\t', lineterminator='\n')
        for item in img_lst:
            writer_.writerow(item)


if __name__ == "__main__":
    main()
