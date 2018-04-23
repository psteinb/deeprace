#!/bin/bash

cd /deeprace/

export HIP_VISIBLE_DEVICES=0

for i in 32 64 128 256 512;
do
	 echo
	 echo $i
	 echo
	 for r in `seq 1 10`;
	 do
	     echo "$0 batch_size=$i run ${r}/10"
	     python3 ./deeprace.py train -c "docker,mi25:1,batch_size=${i}" -t /deeprace/scripts/rocm-short/resnet32v1_bs${i}_rocm_0.tsv -e 15 resnet32v1 > ./scripts/rocm-short/resnet32v1_bs${i}_rocm_${r}.log 2>&1
	 done
done
