#!/usr/bin/env python

"""
@file shuffle_lines.py
@brief shuffle lines in a file
@author ChenglongChen
"""

import sys
import csv
import random
import numpy as np
from string import atoi

def main():

    # collect argvs
    seed = atoi(sys.argv[1])
    file_in = sys.argv[2]
    file_out = sys.argv[3]

    # read
    with open(file_in) as in_:
        lines = in_.readlines()

    # shuffle
    random.seed(seed)
    random.shuffle(lines)

    # write
    with open(file_out, "w") as out_:
        for line in lines:
            out_.write(line)


if __name__ == "__main__":
    main()
