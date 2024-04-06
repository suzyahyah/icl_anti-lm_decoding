#!/usr/bin/env bash
# Author: Suzanna Sia

RUN_MODE=(0 1)

DATAD=$(pwd)/data

for run in ${RUN_MODE[@]}; do
    if [ $run -eq 0 ]; then
    rm -r $DATAD/FLORES
    mkdir -p $DATAD/FLORES #/train
    mkdir -p $DATAD/FLORES/multitask_train
    #mkdir -p $DATAD/FLORES/train
    #mkdir -p $DATAD/FLORES/train

    og_file=https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
    wmt22_supplement=https://dl.fbaipublicfiles.com/flores101/dataset/flores_wmt22_supplement.tar.gz
    wget $og_file -O $DATAD/FLORES/flores101_dataset.tar.gz
    wget $wmt22_supplement -O $DATAD/FLORES/flores_wmt22_supplement.tar.gz

    cd $DATAD/FLORES
    tar zxvf flores101_dataset.tar.gz
    tar zxvf flores_wmt22_supplement.tar.gz

    rm flores101_dataset.tar.gz
    rm flores_wmt22_supplement.tar.gz
    fi

    if [ $run -eq 1 ]; then
        mkdir -p $DATAD/FLORES/flores101_dataset/multitask_train
        dev_dir=$DATAD/FLORES/flores101_dataset/dev
        cp -r $dev_dir  $DATAD/FLORES/flores101_dataset/dev_raw
        fns=$(ls $dev_dir)
        for fn in ${fns[@]}; do
            echo $fn
            sed -n '1,800p' $dev_dir/$fn > $DATAD/FLORES/flores101_dataset/multitask_train/$fn
            sed -n '801,1000p' $dev_dir/$fn > $dev_dir/$fn.tmp
            mv $DATAD/FLORES/flores101_dataset/dev/$fn.tmp $DATAD/FLORES/flores101_dataset/dev/$fn
            cd $DATAD/FLORES/flores101_dataset/multitask_train
            rename .dev .multitask_train *.dev
        done

    fi
done

