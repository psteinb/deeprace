#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:4
#SBATCH -o resnet50-singularity-ngpu4-bs128_%A_%a.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load singularity/2.4.2
TDIR=`mktemp -d`
singularity exec -B $PWD:/home/steinba/deeprace --nv /scratch/steinba/tf1.7-plus.simg python3 /home/steinba/development/deeprace/deeprace.py train -R ${TDIR} -b tf -O n_gpus=4,batch_size=128 -c "k80:4,fs:nfs,singularity:lustre" -t /home/steinba/development/deeprace/scripts/tf-short4/resnet56v1-tf-short-ngpu4-bs128-singularity-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv -e 15 resnet56v1
