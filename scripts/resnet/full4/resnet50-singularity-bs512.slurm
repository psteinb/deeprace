#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:4
#SBATCH -o resnet50-singularity-bs512-ngpu4.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load singularity/2.4.2

singularity exec -B $PWD:/home/steinba/deeprace --nv /scratch/steinba/tf1.5.simg python3 /home/steinba/development/deeprace/deeprace.py train -O n_gpus=4,batch_size=512 -c "k80:4,fs:nfs,singularity:lustre" -t /home/steinba/development/deeprace/scripts/full4/full-resnet56v1-bs512-singularity-ngpu4.tsv resnet56v1
