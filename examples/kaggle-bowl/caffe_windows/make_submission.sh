#!/usr/bin/env bash
# Under Cygwin64

# @file make_submission.sh
# @brief make prediction and submission by averaging a bunch predictions from random transformation
# @author ChenglongChen


##################
## Param Config ##
##################
GLOG_logtostderr=1
# we should have:
# #images = batch_size * ITERATION
TASK=(valid test)
ITERATION=(60 2608)
ENSEMBLE_SIZE=100
LABEL_NUM=121
# see the MODEL_FILE for which index of the desired class probability
# in the output is
SCORE_INDEX=4
GPU_ID=0

for j in $(seq 0 1); do

    #################
    ## Path Config ##
    #################
    # sample submission file
    SAMPLE_FILE=../Data/sampleSubmission.csv
    # image list file
    if [ ${TASK[${j}]} == valid ]; then
        LIST_FILE=../Output/data/list/local_${TASK[${j}]}_list
    else
        LIST_FILE=../Output/data/list/${TASK[${j}]}_list
    fi
    # model prototxt file
    MODEL_FILE=./model/vgg_googlenet_maxout_48x48_${TASK[${j}]}.prototxt
    # set to the best trained model file
    WEIGHT_FILE=../Output/model/vgg_googlenet_maxout_48x48_seed2222_iter_233000.caffemodel
    # probability output file
    PROBA_FILE=../Output/submission/proba_[vgg_googlenet_maxout_48x48]_[ens${ENSEMBLE_SIZE}]_[nll0.78397]_${TASK[${j}]}.csv
    # submission output file
    SUBMISSION_FILE=../Output/submission/submission_[vgg_googlenet_maxout_48x48]_[ens${ENSEMBLE_SIZE}]_[nll0.78397]_${TASK[${j}]}.csv
	
    # use the following to keep track of the logloss and accuracy
    # You can then use the following to see the effect of ensemble:
    # ./utils/plot_caffe_ensemble_logloss.py $LOG_FILE_SINGLE $LOG_FILE_ENSEMBLE
	
    LOG_FILE_SINGLE=${WEIGHT_FILE}.${TASK[${j}]}_logloss_single
    LOG_FILE_ENSEMBLE=${WEIGHT_FILE}.${TASK[${j}]}_logloss_ensemble
    LOG_FILE_SINGLE_ACC=${WEIGHT_FILE}.${TASK[${j}]}_accuracy_single
    LOG_FILE_ENSEMBLE_ACC=${WEIGHT_FILE}.${TASK[${j}]}_accuracy_ensemble
    rm -rf ${LOG_FILE_SINGLE} ${LOG_FILE_ENSEMBLE} ${LOG_FILE_SINGLE_ACC} ${LOG_FILE_ENSEMBLE_ACC}
	
	
    #####################
    ## Make Prediction ##
    #####################
    # get class probability
    for i in $(seq 1 ${ENSEMBLE_SIZE}); do
	
        if [ ${i} == 1 ]; then
	    OUTPUT_FILE=${SUBMISSION_FILE}
	else
            OUTPUT_FILE=${PROBA_FILE}
        fi
	
        # make prediction
        ../../../bin/caffe.exe predict \
                    --model=${MODEL_FILE} \
		    --weights=${WEIGHT_FILE} \
		    --outfile=${OUTPUT_FILE} \
	            --label_number=${LABEL_NUM} \
		    --iterations=${ITERATION[${j}]} \
		    --score_index=${SCORE_INDEX} \
		    --gpu=${GPU_ID} \
		    --random_seed=${i}
		
	# generate kaggle sumission
        ./utils/gen_submission.py ${SAMPLE_FILE} ${LIST_FILE} ${OUTPUT_FILE} ${OUTPUT_FILE}

        # compute logloss and accuracy of this prediction
	echo $"Logloss of the $i prediction: " >> ${LOG_FILE_SINGLE}
        ./utils/compute_logloss.py ${LIST_FILE} ${OUTPUT_FILE} >> ${LOG_FILE_SINGLE}
	echo $"Accuracy of the $i prediction: " >> ${LOG_FILE_SINGLE_ACC}
        ./utils/compute_accuracy.py ${LIST_FILE} ${OUTPUT_FILE} >> ${LOG_FILE_SINGLE_ACC}
		
        # merge prediction
        if [ ${i} != 1 ]; then
            w1=1
            w2=$((i-1))
            ./utils/fuse_submission.py ${OUTPUT_FILE} ${w1} ${SUBMISSION_FILE} ${w2} ${SUBMISSION_FILE}
        fi
	
        # compute logloss and accuracy for the current ensemble
        echo $"Logloss of the current $i ensemble prediction: " >> ${LOG_FILE_ENSEMBLE}
        ./utils/compute_logloss.py ${LIST_FILE} ${SUBMISSION_FILE} >> ${LOG_FILE_ENSEMBLE}
        echo $"Accuracy of the current $i ensemble prediction: " >> ${LOG_FILE_ENSEMBLE_ACC}
        ./utils/compute_accuracy.py ${LIST_FILE} ${SUBMISSION_FILE} >> ${LOG_FILE_ENSEMBLE_ACC}
    done
done