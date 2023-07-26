#!/bin/bash 
#SBATCH -J  job_vox
#SBATCH -o  job_vox.o%j

#SBATCH -A mang
#SBATCH --mail-user=rsultamuratov@uh.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

#SBATCH -t 2-00:00:00
#SBATCH -N 1 -n 2
#SBATCH --gpus=1 
#SBATCH --mem=16GB

module add TensorFlow

# mypython=/home/rsultamu/.conda/envs/tf/bin/python
home=/project/mang/rsultamu/ventricles/voxnet
code=$home/pretrain.py

epochs=$1

cd $home
python $code $epochs