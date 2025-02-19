CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model ULIP_PointBERT \
--npoints 8192 \
--lr 3e-3 \
--pretrain_dataset_name "modelnet40" \
--output-dir ./outputs/reproduce_pointbert_8kpts_version_dataset





CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 main.py \
--model ULIP_PointBERT \
--npoints 8192 \
--lr 1.5e-3 \
--pretrain_dataset_name "modelnet40" \
--output-dir ./outputs/reproduce_pointbert_8kpts_version_dataset



GPUS_PER_NODE=2
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
    --tee 3 \
    main.py \
--model ULIP_PointBERT \
--npoints 8192 \
--lr 1.5e-3 \
--output-dir /data/zhongz2/data/pointllm/outputs/reproduce_pointbert_8kpts_version_dataset


CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model ULIP_PointBERT \
--npoints 8192 \
--lr 1.5e-3 \
--output-dir /data/zhongz2/data/pointllm/outputs/reproduce_pointbert_8kpts_version_dataset



python main.py \
--model ULIP_PointBERT \
--npoints 8192 \
--lr 1.5e-3 \
--output-dir /data/zhongz2/data/pointllm/outputs/reproduce_pointbert_8kpts_version_dataset \
--gpu 0


# shapenetv2

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
    --tee 3 \
     main.py \
--model ULIP2_PointBERT_Colored_1024 \
--npoints 1024 \
--lr 5e-4 \
--batch-size 32 \
--output-dir ./outputs/reproduce_pointbert_1kpts_version_dataset \
--pretrain_dataset_name "shapenetv2"

python main.py \
--model ULIP2_PointBERT_Colored_1024 \
--npoints 1024 \
--lr 1.5e-3 \
--batch-size 2 \
--output-dir ./outputs/reproduce_pointbert_1kpts_version_dataset \
--pretrain_dataset_name "shapenetv2" \
--gpu 0







