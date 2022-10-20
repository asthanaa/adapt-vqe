#!/bin/bash
#SBATCH --job-name=qseh8
#SBATCH --time=144:00:00
#SBATCH --partition=normal_q
#SBATCH --account=nmayhall_group
#SBATCH --nodes=1 
##SBATCH --ntasks=20
#SBATCH --mem=200GB
#SBATCH --exclusive
#SBATCH --output=w100.err
#SBATCH --error=w100.err

source /home/$USER/.bashrc
#export PYTHONPATH=/home/aasthana/adapt/adapt-vqe/:$PYTHONPATH
source activate adapt3

python h8qse.py > 1.5_20_1.5qse  
