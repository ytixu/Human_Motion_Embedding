#!/usr/bin/env bash


loss_func=( "mean_squared_error" "mean_absolute_error" );
optimizers=( "Nadam()" "Adam(lr=0.002)" "Adam()" "Nadam(lr=0.001)" "RMSprop()" );


for loss in "${loss_func[@]}"
do
	for opt in "${optimizers[@]}"
	do
		python train.py -m Seq2Seq -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt;
		python train.py -m VL_RNN -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt;
		python train.py -m H_RNN -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt;
		python train.py -m H_RNN -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt -hs 9 19;
		python train.py -m HH_RNN -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt;
                python train.py -m HH_RNN -t 20 -ti 10 -to 10 -ld 1024 -lf $loss -opt $opt -hs 9 19;
	done
done;

