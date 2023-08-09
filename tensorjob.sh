#!/bin/bash 
#SBATCH -J  tensorjob
#SBATCH -o  tensorjob.o%j

#SBATCH -A mang
#SBATCH --mail-user=rsultamuratov@uh.edu
#SBATCH --mail-type=end
#SBATCH --mail-type=fail

#SBATCH -t 10:00:00
#SBATCH -N 1 -n 2
#SBATCH --gpus=1 
#SBATCH --mem=8GB

module add TensorFlow
cd /project/mang/rsultamu/ventricles
python $@