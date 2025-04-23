#!/bin/bash

#SBATCH --mail-type=FAIL


LR=${1}
BS=${2}
NNODES=${3}
IMG_KEY=${4}

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    cd /scratch/cluster_scratch/zhongz2/ULIP
    DATA_ROOT=/mnt/gridftp/zhongz2
    DST_DIR=/tmp/zhongz2/ULIP
    source /scratch/cluster_scratch/zhongz2/anaconda3/bin/activate th23 #th26
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    module load gcc/11.2.0
else
    cd /home/zhongz2/ULIP
    DATA_ROOT=/data/zhongz2
    DST_DIR=/lscratch/$SLURM_JOB_ID/ULIP
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0  
fi

OUTPUT_DIR=${DATA_ROOT}/ULIP_outputs8/nodes${NNODES}/${IMG_KEY}/${LR}/${BS}
mkdir -p $OUTPUT_DIR
sleep 5

SRC_DIR=${DATA_ROOT}/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video

python generate_train_val.py ${SRC_DIR} ${OUTPUT_DIR}
sleep 1
wait;


if [ ${NNODES} -gt 1 ]; then
echo "multi node"
srun --export ALL --jobid ${SLURM_JOB_ID} bash copydata.sh ${SRC_DIR} ${DST_DIR} ${IMG_KEY} ${OUTPUT_DIR}
else
echo "single node"
bash copydata.sh ${SRC_DIR} ${DST_DIR} ${IMG_KEY} ${OUTPUT_DIR}
fi

wait

if [ ${NNODES} -gt 1 ]; then
srun --export ALL --jobid ${SLURM_JOB_ID} bash train2_v4.sh ${LR} ${BS} ${NNODES} ${IMG_KEY} ${DST_DIR}
else
bash train2_v4.sh ${LR} ${BS} ${NNODES} ${IMG_KEY} ${DST_DIR} ${OUTPUT_DIR}
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


# FRCE
sbatch -p gpu --gres=gpu:4 --cpus-per-task=32 --mem=100G --time=24:00:00 --export=ALL \
train2_v4_main.sh 5e-5 32 1 he

# Biowulf
sbatch --partition=gpu --gres=gpu:v100x:4,lscratch:400 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=100G --time=24:00:00 --export=ALL \
train2_v4_main.sh 5e-5 32 1 he


