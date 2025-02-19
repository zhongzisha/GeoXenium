# source /data/zhongz2/anaconda3/bin/activate pointllm
# module load CUDA/11.8
# module load cuDNN/8.9.2/CUDA-11
# module load gcc/11.3.0

source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0   

# module load cuDNN/8.2.1/CUDA-11.3
export NCCL_ROOT=/data/zhongz2/nccl_2.14.3-1+cuda11.7_x86_64
export NCCL_DIR=/data/zhongz2/nccl_2.14.3-1+cuda11.7_x86_64
export NCCL_PATH=/data/zhongz2/nccl_2.14.3-1+cuda11.7_x86_64
export NCCL_HOME=$NCCL_ROOT
#   export CUDA_HOME=/usr/local/CUDA/11.3.0
# export CUDNN_ROOT=/data/zhongz2/cudnn-11.3-linux-x64-v8.2.0.53
# export CUDNN_PATH=${CUDNN_ROOT}
export PATH=${NCCL_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=INFO

git clone https://github.com/erikwijmans/Pointnet2_PyTorch
git clone https://github.com/unlimblue/KNN_CUDA.git


export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"



    source $FRCE_DATA_ROOT/anaonda3/bin/activate th20
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    module load gcc/11.2.0


srun --export ALL --pty -p quick --gres=gpu:a100:4,lscratch:400 --cpus-per-task=32 --mem=200G  --time=4:00:00 bash



