#!/bin/bash
#SBATCH --job-name=successratesh
#SBATCH --time=24:00:00
#SBATCH --partition=preemptable_q
#SBATCH --account=nmayhall_group
#SBATCH --nodes=1 
#SBATCH --ntasks=4
#SBATCH --mem=1GB
#SBATCH --output=w100.err
#SBATCH --error=w100.err
hostname>w100.err
source /home/$USER/.bashrc

source activate adapt3

python qse.py >out 
