#!/usr/bin/env bash

# python train.py -m HH_RNN -ld 1024 -rep --supervised -iter 500 -gs 3000

loss_func=( "mean_squared_error" "mean_absolute_error" );
optimizers=( "Nadam()" "Adam()" "RMSprop()" );
normalization=( "none" "norm_std" "norm_max");
iters=200;
ld=512;

for loss in "${loss_func[@]}"
do
    for opt in "${optimizers[@]}"
    do
        for norm in "${normalization[@]}"
        do
            python train.py -m Seq2Seq -t 20 -to 10 -ld $ld -loss $loss -opt $opt -w $norm -iter $iters --no_save -gs 30000;
            python train.py -m VL_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -w $norm -iter $iters --no_save -gs 3000;
            # python train.py -m VL_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -iter $iters -rep --no_save;
            # python train.py -m VL_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -hs 9 19 -iter $iters -rep --no_save -gs 30000;
            python train.py -m H_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -w $norm -iter $iters --no_save;
            # python train.py -m H_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -iter $iters -rep --no_save;
            # python train.py -m H_RNN -t 20 -to 10 -ld $ld -loss $loss -opt $opt -hs 9 19 -iter $iters -rep --no_save -gs 30000;
            # python train.py -m HH_RNN -t 30 -to 10 -ld $ld -loss $loss -opt $opt -iter $iters -rep --no_save -gs 30000;
            # python train.py -m HH_RNN -t 30 -to 10 -ut 5 -ld $ld -loss $loss -opt $opt -iter $iters -rep --no_save -gs 30000;
            # python train.py -m H_Seq2Seq -t 30 -to 10 -ld $ld -loss $loss -opt $opt -iter $iters --no_save -gs 30000;
            # python train.py -m H_Seq2Seq -t 30 -to 10 -ut 5 -ld $ld -loss $loss -opt $opt -iter $iters --no_save -gs 30000;
        done;
    done;
done;

#python train.py -m Seq2Seq -ld 1024 --supervised









