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
#SBATCH --mem=4GB

module add TensorFlow
# pip3 install -U scikit-learn

mdl=voxnet

# mypython=/home/rsultamu/.conda/envs/tf/bin/python
home=/project/mang/rsultamu/ventricles/${mdl}
code=$home/train_${mdl}.py
data=/project/mang/rsultamu/ventricles/data

epochs=$1
backbone=$2
classifier=$3
lr=$4

cd $home
python $code $epochs $backbone $classifier $lr