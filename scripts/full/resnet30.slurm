#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o resnet30.log
#SBATCH --mem-per-cpu=8000

cd /home/steinba/development/deeprace/
pwd
module load modenv/both modenv/eb tensorflow/1.3.0-Python-3.5.2 docopt/0.6.2 keras/2.1.4
python3 ./deeprace.py run -c "k80:1,fs:nfs" -t /home/steinba/development/deeprace/scripts/full/full-resnet32v1.tsv resnet32v1
