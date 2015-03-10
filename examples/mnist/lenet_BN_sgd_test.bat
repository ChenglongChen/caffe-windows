SET GLOG_logtostderr=1

"../../bin/MainCaller.exe" test ^
    --model=lenet_BN_train_valid.prototxt ^
	--weights=lenet_iter_10000.caffemodel ^
	--gpu=0
	:2>&1 | "../../tools/wtee.exe" ./lenet_BN_sgd_test.log
	
pause
