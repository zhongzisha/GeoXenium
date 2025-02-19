#!/bin/bash

#SBATCh --mail-type=ALL


source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0  

LR=${1}
BS=${2}

cd /home/zhongz2/ULIP


srun --export ALL --jobid $SLURM_JOB_ID bash copydata.sh

wait

srun --export ALL --jobid $SLURM_JOB_ID bash train2.sh ${LR} ${BS}  

wait
exit;

sbatch --partition=gpu --gres=gpu:a100:4,lscratch:400 --nodes=2 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=24:00:00 --export=ALL \
train2_main.sh 1e-4 32


sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=2 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=4:00:00 --export=ALL \
train2_main.sh 2e-4 32

sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=2 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=4:00:00 --export=ALL \
train2_main.sh 1e-4 32


sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=2 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=4:00:00 --export=ALL \
train2_main.sh 5e-5 32


sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=2 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=4:00:00 --export=ALL \
train2_main.sh 5e-4 32




