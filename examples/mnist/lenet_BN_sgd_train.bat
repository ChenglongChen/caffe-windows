SET GLOG_logtostderr=1

"../../bin/MainCaller.exe" train ^
    --solver=lenet_BN_sgd_solver.prototxt
	:2>&1 | "../../tools/wtee.exe" ./lenet_BN_sgd_train.log
	
pause
