#!/bin/bash

#PBS -q ai
#PBS -N mse_xvector_emb512
#PBS -l select=1:ncpus=8:ngpus=4:mem=128gb
#PBS -l walltime=119:59:00
#PBS -j oe
#PBS -P 12001458

cd $PBS_O_WORKDIR
module load singularity
export SINGULARITY_BIND="/home/project:/home/project,/scratch:/scratch,/app:/app"
singularity run --nv /app/apps/containers/kaldi/kaldi-nvidia-22.04-py3.sif << EOF
source ~/miniconda3/bin/activate ~/miniconda3/envs/wespeaker
python ./GPUtest.py