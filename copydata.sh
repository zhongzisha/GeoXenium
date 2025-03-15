#!/bin/bash


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    cd /scratch/cluster_scratch/zhongz2/ULIP
    DATA_ROOT=/mnt/gridftp/zhongz2
else
    cd /home/zhongz2/ULIP
    DATA_ROOT=/data/zhongz2
fi


set -x -e
echo "copy data on `hostname`"


IMG_KEY=${1}
DST_DIR=${2}

SRC_DIR=${DATA_ROOT}/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version7_with_video


for subset in "train" "val"; do

mkdir -p ${DST_DIR}/${subset}

bash extract_tars.sh $SRC_DIR/${subset}_list_${IMG_KEY}.txt ${SRC_DIR} ${DST_DIR}/${subset}
# cp $SRC_DIR/../X_pca_3.npy $DST_DIR/${subset}
ln -sf $SRC_DIR/../cell_feature_matrix.h5 $DST_DIR/${subset}
cp $SRC_DIR/${subset}_items.pkl $DST_DIR/${subset}
cp $SRC_DIR/${subset}_items.csv $DST_DIR/${subset}
done


exit;










