#!/bin/bash

cd /deeprace/

export HIP_VISIBLE_DEVICES=0

for r in `seq 1 10`;
do

     for i in 32 64 128 256 512;
	 do
         if [ -e /deeprace/scripts/rocm-short/resnet32v1_bs${i}_rocm_${r}.tsv ];then
             echo "skipping batch_size=$i run ${r}/10 as it already exists"
             continue
         else
	         echo "$0 batch_size=$i run ${r}/10"
	         python3 ./deeprace.py train -O "batch_size=${i}" -c "docker,mi25:1" -t /deeprace/scripts/rocm-short/resnet32v1_bs${i}_rocm_${r}.tsv -e 10 resnet32v1 > ./scripts/rocm-short/resnet32v1_bs${i}_rocm_${r}.log 2>&1
         fi
	 done
done
