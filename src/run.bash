#!/usr/bin/env bash

python train.py -m HH_RNN -ld 1024 -rep --supervised -iter 500

loss_func=( "mean_squared_error" "mean_absolute_error" );
optimizers=( "Nadam()" "Adam(lr=0.002)" "Adam()" "Nadam(lr=0.001)" "RMSprop()" );
iters=200;
ld=500;

for loss in "${loss_func[@]}"
do
	for opt in "${optimizers[@]}"
	do
		python train.py -m Seq2Seq -t 20 -to 10 -ld $ld -lf $loss -opt $opt -iter $iters;
		python train.py -m VL_RNN -t 20 -to 10 -ld $ld -lf $loss -opt $opt -iter $iters;
		python train.py -m H_RNN -t 20 -to 10 -ld $ld -lf $loss -opt $opt -iter $iters;
		python train.py -m H_RNN -t 20 -to 10 -ld $ld -lf $loss -opt $opt -hs 9 19 -iter $iters;
		python train.py -m HH_RNN -t 20 -to 10 -ld $ld -lf $loss -opt $opt -iter $iters;
		python train.py -m HH_RNN -t 20 -to 10 -ld $ld -lf $loss -opt $opt -hs 9 19 -iter $iters;
	done
done;

#python train.py -m Seq2Seq -ld 1024 --supervised

