#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o resnet50-tf-short-singularity_%A_%a.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load singularity/2.4.2

singularity exec -B $PWD:/home/steinba/deeprace --nv /scratch/steinba/tf1.7-plus.simg python3 /home/steinba/development/deeprace/deeprace.py train -R `mktemp -d` -b tf -c "k80:1,fs:nfs,singularity:lustre" -t /home/steinba/development/deeprace/scripts/tf-short/resnet56v1-tf-short-singularity-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv -e 15 resnet56v1
