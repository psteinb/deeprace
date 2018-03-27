#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o resnet50-bs512_%A_%a.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load modenv/both modenv/eb tensorflow/1.3.0-Python-3.5.2 docopt/0.6.2 keras/2.1.4
python3 ./deeprace.py run -O batch_size=512 -c "k80:1,fs:nfs" -t /home/steinba/development/deeprace/scripts/full/full-resnet56v1-bs512-${SLURM_ARRAY_TASK_ID}.tsv resnet56v1
