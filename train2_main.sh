#!/bin/bash

#SBATCh --mail-type=ALL


LR=${1}
BS=${2}
NNODES=${3}

cd /home/zhongz2/ULIP

if [ ${NNODES} -gt 1 ]; then
echo "multi node"
srun --export ALL --jobid ${SLURM_JOB_ID} bash copydata.sh
else
echo "single node"
bash copydata.sh
fi

wait

if [ ${NNODES} -gt 1 ]; then
srun --export ALL --jobid ${SLURM_JOB_ID} bash train2.sh ${LR} ${BS} ${NNODES}  
else
bash train2.sh ${LR} ${BS} ${NNODES}
fi

wait
exit;


for LR in 1e-4 1e-5; do
sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=4:00:00 --export=ALL \
train2_main.sh ${LR} 32 1
done


