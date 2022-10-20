#!/bin/bash
#SBATCH --job-name=adapth4h4
#SBATCH --time=120:00:00
#SBATCH --partition=normal_q
#SBATCH --account=nmayhall_group
#SBATCH --nodes=1 
#SBATCH --ntasks=20
#SBATCH --mem=80GB
#SBATCH --output=w100.err
#SBATCH --error=w100.err

source /home/$USER/.bashrc
#export PYTHONPATH=/home/aasthana/adapt/adapt-vqe/:$PYTHONPATH
conda activate adapt3

#python qse_symbreak.py>out_symbrk2
python qse.py>out2
