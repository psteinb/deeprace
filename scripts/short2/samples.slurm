#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH -n 1
#SBATCH -p gpu2
#SBATCH --gres=gpu:1
#SBATCH -o test.log
#SBATCH --mem-per-cpu=8000

module load singularity/2.4.2
hostname

nvidia-smi

/home/steinba/development/nvidia-samples/9.0.176/1_Utilities/deviceQuery/deviceQuery

singularity exec --nv /scratch/steinba/tf1.5.simg /home/steinba/development/nvidia-samples/9.0.176/1_Utilities/deviceQuery/deviceQuery
