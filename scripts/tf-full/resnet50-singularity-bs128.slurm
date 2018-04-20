#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o resnet50-singularity-bs128.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load singularity/2.4.2

singularity exec -B $PWD:/home/steinba/deeprace --nv /scratch/steinba/tf1.7-plus.simg python3 /home/steinba/development/deeprace/deeprace.py train -b tf -O batch_size=128 -c "k80:1,fs:nfs,singularity:lustre" -t /home/steinba/development/deeprace/scripts/tf-full/tf-full-resnet56v1-bs128-singularity.tsv resnet56v1
