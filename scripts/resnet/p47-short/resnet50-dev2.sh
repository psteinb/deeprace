#!/bin/bash

cd /deeprace/

export HIP_VISIBLE_DEVICES=2

for rep in `seq 1 10`;

do
    for bs in 64 128;
	do
        if [ -e /deeprace/scripts/rocm-short/resnet56v1_bs${bs}_rocm_${rep}.tsv ];then
            echo "skipping batch_size=${bs} run ${rep}/10"
            continue
        else
	     echo "$0 batch_size=${bs} run ${rep}/10"
	     python3 ./deeprace.py train -O "batch_size=${bs}" -c "docker,mi25:1" -t /deeprace/scripts/rocm-short/resnet56v1_bs${bs}_rocm_${rep}.tsv -e 10 resnet56v1 > ./scripts/rocm-short/resnet56v1_bs${bs}_rocm_${rep}.log 2>&1
        fi
	 done
done
