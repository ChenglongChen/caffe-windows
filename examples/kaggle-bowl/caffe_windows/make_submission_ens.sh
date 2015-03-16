#!/usr/bin/env bash
# Under Cygwin64

# @file make_submission.sh
# @brief make greedy bagging ensemble
# @author ChenglongChen

LIST_FILE=../Output/data/list/local_valid_list
# set to submission file without the "_valid.csv" or "_test.csv"
SUBMISSION_FILE1=../Output/submission/submission1
SUBMISSION_FILE2=../Output/submission/submission2
SUBMISSION_FILE3=../Output/submission/submission3
# output ensemble submission (the ensemble logloss and ".csv" will be padded in the end of the file name)
SUBMISSION_FILE_ENS=../Output/submission/submission_greedy_ensemble
	
# mode can be: 
# - greedy (greedy bagging ensemble)
# - average (naive average)
mode=greedy
# task can be: 
# - valid (if you don't want to write the ensemble file for test or you currently don't have test version for each SUBMISSION_FILE's)
# - test (also write the ensemble for test)
task=test
./utils/greedy_bagging_ensemble.py ${LIST_FILE} ${SUBMISSION_FILE_ENS} ${mode} ${task} ${SUBMISSION_FILE1} ${SUBMISSION_FILE2} ${SUBMISSION_FILE3}