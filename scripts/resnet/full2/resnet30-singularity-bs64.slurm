#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:2
#SBATCH -o resnet30-singularity-ngpu2.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load singularity/2.4.2

singularity exec -B $PWD:/home/steinba/deeprace --nv /scratch/steinba/tf1.5.simg python3 /home/steinba/development/deeprace/deeprace.py train -O n_gpus=2,batch_size=64 -c "k80:2,fs:nfs,singularity:lustre" -t /home/steinba/development/deeprace/scripts/full2/full-resnet32v1-bs64-singularity-ngpu2.tsv resnet32v1
