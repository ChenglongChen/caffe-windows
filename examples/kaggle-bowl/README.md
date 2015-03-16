# Kaggle's National Data Science Bowl

Example using Caffe with realtime data augmentation for [Kaggle's National Data Science Bowl](https://www.kaggle.com/c/datasciencebowl).

You should get a logloss about ~0.67xx.

## Instruction
* download data from the [competition website](https://www.kaggle.com/c/datasciencebowl/data) and put all the data into `./Data` dir. You should have
 - `./Data/train`
 - `./Data/test`
 - `./Data/sampleSubmission.csv`
* run `./caffe_windows/prepare_data.sh` to convert data into Caffe accepted format.
* run `./caffe_windows/train_caffenet.bat` to train Caffe model. Besides the model, it will also save the screen print into a log file which will be used in the next step.
* use `./caffe_windows/utils/plot_caffe_training_logloss.py` to plot the training and validation logloss (from the saved log file) and then pick up the best model.
* (optional) you can try to finetune the best model by 1) lowering the loss_weight for the coarse classifier, 2) lowering the learning rate, 3) decreasing the augmentation (trick from Dr. Benjamin Graham in the [winning solution for Cifar10](https://www.kaggle.com/c/cifar-10/forums/t/10493/train-you-very-own-deep-convolutional-network/56267)).
* run `./caffe_windows/make_submission.sh` to generate Kaggle submission by averaging the predictions from random augmentation. Be sure you set the weight file to the best model in the script.
* use `./caffe_windows/utils/plot_caffe_ensemble_logloss.py` to see the effect of averaging predictions.

## Additional Info

1. Make sure you set the right path in the above file.
2. For execution of .sh script under Windows, I use Cygwin64.
3. There is greedy bagging ensemble (and also naive averaging) for ensembling predictions based on the validation logloss. Have a look at `./caffe_windows/make_submission_ens.sh` and `./caffe_windows/utils/greedy_bagging_ensemble.py`.