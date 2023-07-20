#!/bin/bash 
#SBATCH -J  job_point
#SBATCH -o  job_point.o%j

#SBATCH -A mang
#SBATCH --mail-user=rsultamuratov@uh.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

#SBATCH -t 2-00:00:00
#SBATCH -N 1 -n 4
#SBATCH --gpus=1 
#SBATCH --mem=16GB

module add TensorFlow
# pip3 install -U scikit-learn

mdl=pointnet

# mypython=/home/rsultamu/.conda/envs/tf/bin/python
home=/project/mang/rsultamu/ventricles/${mdl}
code=$home/train_${mdl}.py
data=/project/mang/rsultamu/ventricles/data

epochs=$1
backbone=$2
classifier=$3
jit=$4

cd $home
python $code $epochs $backbone $classifier $jit