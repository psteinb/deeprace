#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o resnet50.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load modenv/both modenv/eb tensorflow/1.3.0-Python-3.5.2 docopt/0.6.2 keras/2.1.4 h5py/2.6.0-intel-2016.03-GCC-5.3-Python-3.5.2-HDF5-1.8.17-serial
python3 ./deeprace.py train -c "k80:1,fs:nfs" -t /home/steinba/development/deeprace/scripts/full/full-resnet56v1.tsv resnet56v1
