#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:4
#SBATCH -o resnet30-bs512_%A_%a-ngpu4.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load modenv/both modenv/eb tensorflow/1.3.0-Python-3.5.2 docopt/0.6.2 keras/2.1.4 h5py/2.6.0-intel-2016.03-GCC-5.3-Python-3.5.2-HDF5-1.8.17-serial
python3 ./deeprace.py train -O n_gpus=2,batch_size=512 -c "k80:4,fs:nfs" -t /home/steinba/development/deeprace/scripts/full4/full-resnet32v1-bs512-${SLURM_ARRAY_TASK_ID}-ngpu4.tsv resnet32v1
