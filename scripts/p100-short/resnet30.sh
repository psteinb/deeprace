#!/bin/bash

cd /home/steinba/development/deeprace/scripts/p100-short
export CUDA_VISIBLE_DEVICES=1

module load singularity/2.4.2

for r in `seq 1 10`;
do
	 echo
	 echo $i
	 echo
	 for i in 32 64 128 256 512;
	 do
	     echo "$0 batch_size=$i run ${r}/10"
	     singularity exec -B /home/steinba/development/deeprace/:/deeprace --nv /scratch/steinba/tf1.3.simg python3 /deeprace/deeprace.py train -O batch_size=${i} -c "p100:1,fs:nfs,singularity:lustre" -t /deeprace/scripts/p100-short/resnet32v1-short-bs${i}-singularity-${r}.tsv -e 10 resnet32v1

	     singularity exec -B /home/steinba/development/deeprace/:/deeprace --nv /scratch/steinba/tf1.7.simg python3 /deeprace/deeprace.py train -O batch_size=${i} -b tf -c "p100:1,fs:nfs,singularity:lustre" -t /deeprace/scripts/p100-short/resnet32v1-short-bs${i}-singularity-${r}.tsv -e 10 resnet32v1

	 done
done
