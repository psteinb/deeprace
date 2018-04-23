#!/bin/bash

cd /home/steinba/development/deeprace/scripts/gtx1080-short
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
	     singularity exec -B /home/steinbac/development/deeprace/:/deeprace --nv /projects/steinbac/software/singularity/sandbox//tf1.3-plus.simg python3 /deeprace/deeprace.py train -O batch_size=${i} -c "gtx1080:1,fs:nfs,singularity:lustre" -t /deeprace/scripts/gtx1080-short/resnet32v1-short-bs${i}-singularity-${r}.tsv -e 10 resnet32v1

	     singularity exec -B /home/steinbac/development/deeprace/:/deeprace --nv /projects/steinbac/software/singularity/sandbox//tf1.7-plus.simg python3 /deeprace/deeprace.py train -O batch_size=${i} -b tf -c "gtx1080:1,fs:nfs,singularity:lustre" -t /deeprace/scripts/gtx1080-short/resnet32v1-short-bs${i}-singularity-${r}.tsv -e 10 resnet32v1

	 done
done

export CUDA_VISIBLE_DEVICES=0,1

for r in `seq 1 10`;
do
	 echo
	 echo $i
	 echo
	 for i in 32 64 128 256 512;
	 do
	     echo "$0 batch_size=$i run ${r}/10"
	     singularity exec -B /home/steinbac/development/deeprace/:/deeprace --nv /projects/steinbac/software/singularity/sandbox//tf1.3-plus.simg python3 /deeprace/deeprace.py train -O batch_size=${i},n_gpus=2 -c "gtx1080:2,fs:nfs,singularity:lustre" -t /deeprace/scripts/gtx1080-short/resnet32v1-short-bs${i}-2gpus-singularity-${r}.tsv -e 10 resnet32v1

	     singularity exec -B /home/steinbac/development/deeprace/:/deeprace --nv /projects/steinbac/software/singularity/sandbox//tf1.7-plus.simg python3 /deeprace/deeprace.py train -O batch_size=${i},n_gpus=2 -b tf -c "gtx1080:2,fs:nfs,singularity:lustre" -t /deeprace/scripts/gtx1080-short/resnet32v1-short-bs${i}-2gpus-singularity-${r}.tsv -e 10 resnet32v1

	 done
done
