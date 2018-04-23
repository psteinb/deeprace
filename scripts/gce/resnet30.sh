#!/bin/bash
singularity exec -B /home/steinbac/deeprace:/deeprace --nv /home/steinbac/tf1.3-plus.simg nvidia-smi
#singularity exec -B /home/steinbac/deeprace:/deeprace --nv /home/steinbac/tf1.3-plus.simg python /deeprace/deeprace.py train -O batch_size=${i} -c "k80:1,fs:gce,singularity:gce" -t /home/steinbac/resnet32v1-short-gce-bs128-singularity_0.tsv -e 15 resnet32v1
for r in `seq 1 10`;
do
	 echo
	 echo $i
	 echo
     for i in 512 265 128 64 32;
	 do
	     if [[ -e /home/steinbac/resnet32v1-short-gce-bs${i}-singularity_${r}.tsv ]];then
             echo "$0 batch_size=$i run ${r}/10 already exists"
             continue;
         else
             echo "$0 batch_size=$i run ${r}/10"
             singularity exec -B /home/steinbac/deeprace:/deeprace --nv /home/steinbac/tf1.3-plus.simg python /deeprace/deeprace.py train -O batch_size=${i} -c "k80:1,fs:gce,singularity:gce" -t /home/steinbac/resnet32v1-short-gce-bs${i}-singularity_${r}.tsv -e 10 resnet32v1
         fi
	 done
done
