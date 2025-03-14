#!/bin/bash



set -x -e
echo "copy data on `hostname`"

cd /home/zhongz2/ULIP

IMG_KEY=${1}

SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version3
SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version4
SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version5
SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version6
DST_DIR=/lscratch/$SLURM_JOB_ID

for subset in "train" "val"; do

mkdir -p ${DST_DIR}/${subset}

bash extract_tars.sh $SRC_DIR/${subset}_list_${IMG_KEY}.txt ${SRC_DIR} ${DST_DIR}/${subset}
# cp $SRC_DIR/../X_pca_3.npy $DST_DIR/${subset}
ln -sf $SRC_DIR/../cell_feature_matrix.h5 $DST_DIR/${subset}
cp $SRC_DIR/${subset}_items.pkl $DST_DIR/${subset}

done


exit;










