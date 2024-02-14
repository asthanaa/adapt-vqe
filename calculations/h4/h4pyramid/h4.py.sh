#!/bin/bash
#SBATCH --job-name=h4.py.sh
#SBATCH --time=48:00:00
#SBATCH --partition=normal_q
#SBATCH --account=nmayhall_group
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --output=h4.err
#SBATCH --error=h4.err
hostname > h4.err

source /home/$USER/.bashrc

source activate adapt3

cd $TMPDIR
mkdir h4.py20220728132512
cp /home/aasthana/adapt/adapt-vqe/h4/h4pyramid/h4.py h4.py
nohup python h4.py > h4.out20220728132512
cp h4.out20220728132512 /home/aasthana/adapt/adapt-vqe/h4/h4pyramid/h4.out
cat h4.err20220728132512 >> /home/aasthana/adapt/adapt-vqe/h4/h4pyramid/h4.err
rm -rf h4.py20220728132512*
rm  h4.py*
rm h4.out20220728132512
rm h4.err20220728132512
