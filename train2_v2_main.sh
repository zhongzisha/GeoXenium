#!/bin/bash

#SBATCh --mail-type=ALL


LR=${1}
BS=${2}
NNODES=${3}
IMG_KEY=${4}

cd /home/zhongz2/ULIP

if [ ${NNODES} -gt 1 ]; then
echo "multi node"
srun --export ALL --jobid ${SLURM_JOB_ID} bash copydata.sh ${IMG_KEY}
else
echo "single node"
bash copydata.sh ${IMG_KEY}
fi

wait

if [ ${NNODES} -gt 1 ]; then
srun --export ALL --jobid ${SLURM_JOB_ID} bash train2_v2.sh ${LR} ${BS} ${NNODES} ${IMG_KEY} 
else
bash train2_v2.sh ${LR} ${BS} ${NNODES} ${IMG_KEY}
fi

wait
exit;


IMG_KEY="he"
for LR in 5e-5 1e-5 1e-4; do
sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=200G --time=4:00:00 --export=ALL \
train2_v2_main.sh ${LR} 32 1 ${IMG_KEY}
sleep 1
done



IMG_KEY="dapi"
for LR in 5e-5 1e-5 1e-4; do
sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=200G --time=4:00:00 --export=ALL \
train2_v2_main.sh ${LR} 32 1 ${IMG_KEY}
sleep 1
done
