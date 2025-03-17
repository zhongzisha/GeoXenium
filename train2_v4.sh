#!/bin/bash


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    cd /scratch/cluster_scratch/zhongz2/ULIP
    DATA_ROOT=/mnt/gridftp/zhongz2

    source /data/zhongz2/anaconda3/bin/activate th26
else
    cd /home/zhongz2/ULIP
    DATA_ROOT=/data/zhongz2

    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0  
fi

LR=${1}
BS=${2}
NNODES=${3}
IMG_KEY=${4}
DST_DIR=${5}

OUTPUT_DIR=${DATA_ROOT}/ULIP_outputs6/nodes${NNODES}/${IMG_KEY}/${LR}/${BS}
OUTPUT_DIR=${DATA_ROOT}/ULIP_outputs8/nodes${NNODES}/${IMG_KEY}/${LR}/${BS}
mkdir -p $OUTPUT_DIR
sleep 5

if [ "${SLURM_JOB_NODELIST}" != "" ]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    NNODES=$SLURM_NNODES
    GPUS_PER_NODE=4
else
    MASTER_ADDR=`hostname`
    NNODES=1
    GPUS_PER_NODE=4
fi
MASTER_PORT=25199

echo "begin training on `hostname`, NNODES: ${NNODES}, MASTER: ${MASTER_ADDR}"


num=$(ls ${OUTPUT_DIR}/log*.txt | wc -l)
num=$(($num+1))

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
     main_v4.py \
--model ULIP2_PointBERT_Colored_1024_NoText_NoGeneLoss \
--npoints 1024 \
--lr ${LR} \
--batch-size ${BS} \
--output-dir ${OUTPUT_DIR} \
--pretrain_dataset_name "shapenetv2" \
--pretrain_dataset_prompt "shapenetv2_64" \
--validate_dataset_name "shapenetv2" \
--validate_dataset_prompt "shapenetv2_64" \
--data-path ${DST_DIR} \
--img-key ${IMG_KEY} \
--input-dim 5009 \
2>&1 | tee ${OUTPUT_DIR}/log${num}.txt


exit;




torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --rdzv_endpoint localhost:24999 \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
    --tee 3 \
     main_v4.py \
--model ULIP2_PointBERT_Colored_1024_NoText_NoGeneLoss \
--npoints 1024 \
--lr 5e-5 \
--batch-size 32 \
--output-dir /lscratch/$SLURM_JOB_ID/outputs \
--pretrain_dataset_name "shapenetv2" \
--pretrain_dataset_prompt "shapenetv2_64" \
--validate_dataset_name "shapenetv2" \
--validate_dataset_prompt "shapenetv2_64" \
--data-path /lscratch/$SLURM_JOB_ID \
--img-key he \
--input-dim 5009


