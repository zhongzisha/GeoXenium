#!/bin/bash


#SBATCH --mail-type=FAIL




source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0  

LR=${1}
BS=${2}

cd /home/zhongz2/ULIP

SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version2
DST_DIR=/lscratch/$SLURM_JOB_ID
for f in `ls ${SRC_DIR}/patches*.tar.gz`; do 
tar -xf $f -C $DST_DIR/;
done
cp $SRC_DIR/X_pca_3.npy $DST_DIR/
cp $SRC_DIR/train_items.pkl $DST_DIR/
ln data/

sleep 1;

OUTPUT_DIR=/data/zhongz2/ULIP_outputs/gpu/${LR}/${BS}

GPUS_PER_NODE=4
NNODES=1
MASTER_ADDR=localhost
MASTER_PORT=25199
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --role `hostname -s`: \
     main.py \
--model ULIP2_PointBERT_Colored_1024 \
--npoints 1024 \
--lr ${LR} \
--batch-size ${BS} \
--output-dir ${OUTPUT_DIR} \
--pretrain_dataset_name "shapenetv2" \
--data-path $DST_DIR

exit;

sbatch --partition=quick --time=4:00:00 --cpus-per-task=32 --mem=100G --gres=gpu:a100:4,lscratch:400 \
train.sh 1e-4 32



















for LR in 1e-3 5e-4 1e-4 5e-5; do
for BS in 32; do
sbatch --partition=quick --time=4:00:00 --cpus-per-task=32 --mem=100G --gres=gpu:a100:4,lscratch:400 \
train.sh ${LR} ${BS}
done
done

for LR in 1e-5 5e-6 1e-6; do
for BS in 32; do
sbatch --partition=quick --time=4:00:00 --cpus-per-task=32 --mem=100G --gres=gpu:a100:4,lscratch:400 \
train.sh ${LR} ${BS}
done
done









