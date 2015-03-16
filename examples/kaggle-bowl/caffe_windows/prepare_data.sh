#!/usr/bin/env bash
# Under Cygwin64

# @file prepare_data.sh
# @brief prepares data for training caffenet
# @author ChenglongChen


#################
## Path Config ##
#################
echo "Set path"
# version of handcrafted feature
VERSION=1
CAFFE_TOOL=D:/ChenglongChen/DL/caffe_accum_grad_multiscale/bin
NDSB_PATH=D:/ChenglongChen/DL/caffe_accum_grad_multiscale/examples/kaggle-bowl
DATA_ROOT=$NDSB_PATH/Data
SAMPLE_SUBMISSION=$DATA_ROOT/sampleSubmission.csv
OUTPUT_ROOT=$NDSB_PATH/Output/data
mkdir $OUTPUT_ROOT
mkdir $NDSB_PATH/Output/model
mkdir $NDSB_PATH/Output/submission
mkdir $OUTPUT_ROOT/list
mkdir $OUTPUT_ROOT/feat
mkdir $OUTPUT_ROOT/hdf5
mkdir $OUTPUT_ROOT/leveldb/compact
mkdir $OUTPUT_ROOT/leveldb/normal
echo "Done"


#########################
## Generate Image List ##
#########################
echo "Generate Image List"
cd utils
TASK=(train test)
IMAGE_FOLDER=(train test)
for i in $(seq 0 1); do
    ./gen_image_list.py ${TASK[$i]} \
        $SAMPLE_SUBMISSION \
        $DATA_ROOT/${IMAGE_FOLDER[$i]}/ \
        $OUTPUT_ROOT/list/${TASK[$i]}_list
done
cd ..
echo "Done"


##################################
## Generate Handcrafted Feature ##
##################################
echo "Generate Handcrafted Feature"
cd preprocess
TASK=(train test)
IMAGE_FOLDER=(train test)
for i in $(seq 0 1); do
    ./gen_image_feature.py ${VERSION} ${TASK[$i]} \
        $DATA_ROOT/${IMAGE_FOLDER[$i]}/ \
        $OUTPUT_ROOT/list/${TASK[$i]}_list \
        $OUTPUT_ROOT/feat/${TASK[$i]}_feat
done
cd ..
echo "Done"

#<<BEGIN
#######################
## Train-Valid Split ##
#######################
# shuffle list, feat, and split into local-training and local-validation set
# the SAME seed MUST be used to synchronize the _list and _feat
SEED1=1234
SEED2=5678
which=(list feat)
NUMVALID=3000
NUMTRAIN=$((30336-${NUMVALID}))
cd utils
for i in $(seq 0 1); do
    # copy for backup
    cp $OUTPUT_ROOT/${which[$i]}/train_${which[$i]} $OUTPUT_ROOT/${which[$i]}/train_${which[$i]}_backup
	
    # shuffle it and store in a tmp file
    ./shuffle_lines.py $SEED1 \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]}_backup \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]}
		
    # local-training set
    head -n ${NUMTRAIN} \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]} > \
        $OUTPUT_ROOT/${which[$i]}/local_train_${which[$i]}
		
    # local-validation set
    tail -n +$((${NUMTRAIN}+1)) \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]} > \
        $OUTPUT_ROOT/${which[$i]}/local_valid_${which[$i]}
		
    # shuffle it again for the final finetuning
    ./shuffle_lines.py $SEED2 \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]}_backup \
        $OUTPUT_ROOT/${which[$i]}/train_${which[$i]}
done
cd ..


##########################
## Handcrafted --> HDF5 ##
##########################
echo "Put Handcrafted Feature into HDF5"
CHUNK_SIZE=10000
cd preprocess
TASK=(local_train local_valid train test)
for i in $(seq 0 3); do
    ./gen_hdf5_feat.py $DATA_ROOT/sampleSubmission.csv \
        $OUTPUT_ROOT/feat/${TASK[$i]}_feat \
        $OUTPUT_ROOT/hdf5/${TASK[$i]}_hdf5 \
        $CHUNK_SIZE ${TASK[$i]}
done
cd ..
echo "Done"


##############################
## Generate Compact Leveldb ##
##############################
echo "Generate Compact Leveldb"
TASK=(local_train local_valid train test)
IMAGE_FOLDER=(train train train test)
for i in $(seq 0 3); do
    rm -rf $OUTPUT_ROOT/leveldb/compact/${TASK[$i]}_leveldb_compact
    echo "Creating $OUTPUT_ROOT/leveldb/compact/${TASK[$i]}_leveldb_compact"
    GLOG_logtostderr=1 $CAFFE_TOOL/convert_imageset_compact.exe \
        $DATA_ROOT/${IMAGE_FOLDER[$i]}/ \
        $OUTPUT_ROOT/list/${TASK[$i]}_list \
        $OUTPUT_ROOT/leveldb/compact/${TASK[$i]}_leveldb_compact
done
echo "Done"


#########################
## Generate Mean Image ##
#########################
# generate the normal version for computing the mean file of varying size
echo "Generate Mean Image"
CROP_H=(48)
CROP_W=(48)
TASK=(local_train)
IMAGE_FOLDER=(train)
for i in $(seq 0 0); do
    for j in $(seq 0 0); do
        # create normal leveldb dataset	
        rm -rf $OUTPUT_ROOT/leveldb/normal/${TASK[$i]}_leveldb_${CROP_H[$j]}x${CROP_W[$j]}
        echo "Creating $OUTPUT_ROOT/leveldb/normal/${TASK[$i]}_leveldb_${CROP_H[$j]}x${CROP_W[$j]}"
        GLOG_logtostderr=1 $CAFFE_TOOL/convert_imageset.exe \
            --resize_height=${CROP_H[$j]} \
            --resize_width=${CROP_W[$j]} \
            --gray \
            $DATA_ROOT/${IMAGE_FOLDER[$i]}/ \
            $OUTPUT_ROOT/list/${TASK[$i]}_list \
            $OUTPUT_ROOT/leveldb/normal/${TASK[$i]}_leveldb_${CROP_H[$j]}x${CROP_W[$j]}
			
	# compute image mean
	$CAFFE_TOOL/compute_image_mean.exe \
            $OUTPUT_ROOT/leveldb/normal/${TASK[$i]}_leveldb_${CROP_H[$j]}x${CROP_W[$j]} \
            $OUTPUT_ROOT/leveldb/normal/${TASK[$i]}_mean_${CROP_H[$j]}x${CROP_W[$j]}.binaryproto
    done	
done
echo "Done"

echo "All Done"
echo "Now run train_caffenet.bat to train caffenet"