#!/bin/bash

#SBATCh --mail-type=ALL


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    cd /scratch/cluster_scratch/zhongz2/ULIP
    DATA_ROOT=/mnt/gridftp/zhongz2
    DST_DIR=/tmp/zhongz2/ULIP
else
    cd /home/zhongz2/ULIP
    DATA_ROOT=/data/zhongz2
    DST_DIR=/lscratch/$SLURM_JOB_ID/ULIP
fi


LR=${1}
BS=${2}
NNODES=${3}
IMG_KEY=${4}


if [ ${NNODES} -gt 1 ]; then
echo "multi node"
srun --export ALL --jobid ${SLURM_JOB_ID} bash copydata.sh ${IMG_KEY} ${DST_DIR}
else
echo "single node"
bash copydata.sh ${IMG_KEY} ${DST_DIR}
fi

wait

if [ ${NNODES} -gt 1 ]; then
srun --export ALL --jobid ${SLURM_JOB_ID} bash train2_v4.sh ${LR} ${BS} ${NNODES} ${IMG_KEY} ${DST_DIR}
else
bash train2_v4.sh ${LR} ${BS} ${NNODES} ${IMG_KEY} ${DST_DIR}
fi

wait
exit;


IMG_KEY="he"
for LR in 5e-5 1e-5 1e-4; do
sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=200G --time=4:00:00 --export=ALL \
train2_v4_main.sh ${LR} 32 1 ${IMG_KEY}
sleep 1
done



IMG_KEY="dapi"
for LR in 5e-5 1e-5 1e-4; do
sbatch --partition=quick --gres=gpu:a100:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=200G --time=4:00:00 --export=ALL \
train2_v4_main.sh ${LR} 32 1 ${IMG_KEY}
sleep 1
done
