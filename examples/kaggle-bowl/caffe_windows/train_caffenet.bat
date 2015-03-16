: Under cmd

: @file train_caffenet.bat
: @brief train caffenet
: @author ChenglongChen

SET GLOG_logtostderr=1
SET UTILS_ROOT=D:/ChenglongChen/DL/caffe_accum_grad_multiscale/examples/kaggle-bowl/caffe_windows/utils

SET seed=2222
for /f "tokens=*" %%j in ('%UTILS_ROOT%/date.exe +"%%y%%m%%d_%%H%%M"') do set timestamp=%%j
SET solver_file=./model/vgg_googlenet_maxout_48x48_seed%seed%_solver.prototxt
SET log_file=./model/vgg_googlenet_maxout_48x48_seed%seed%_%timestamp%.log
: disconnect windows remote connect for using GPU
: for %%k in (0 1 2 3 4 5 6 7 8 9) do (tscon %%k /dest:console)
: train caffemodel
"../../../bin/caffe.exe" train ^
    --solver=%solver_file% ^
    2>&1 | "%UTILS_ROOT%/wtee.exe" %log_file%

: You can then use the following to keep track of the training precedure and pick up the best model
: python ./utils/plot_caffe_training_logloss.py %log_file%

pause