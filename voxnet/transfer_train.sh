#!/bin/bash 
#SBATCH -J  job_transfer_vox
#SBATCH -o  job_transfer_vox.o%j

#SBATCH -A mang
#SBATCH --mail-user=rsultamuratov@uh.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

#SBATCH -t 2-00:00:00
#SBATCH -N 1 -n 2
#SBATCH --gpus=1 
#SBATCH --mem=8GB

module add TensorFlow

# mypython=/home/rsultamu/.conda/envs/tf/bin/python
home=/project/mang/rsultamu/ventricles
code=$home/voxnet/transfer_train.py

epochs=$1
cls=$2
lr=$3

cd $home
python $code $epochs $cls $lr