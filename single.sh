#!/usr/bin/bash

module load gcc/8.2.0 python_gpu/3.9.9

export HTTP_PROXY=http://proxy.ethz.ch:3128
export HTTPs_PROXY=http://proxy.ethz.ch:3128

sbatch --time 4-0 --ntasks=4 --mem-per-cpu=8G --gpus=rtx_2080_ti:1  --wrap "PYTHONPATH=/cluster/home/bgunders/deeplearningproject /cluster/scratch/bgunders/conda_envs/sg3exl/bin/python3.9 main.py --dataset cifar10 --variant mixup --epochs 90"

