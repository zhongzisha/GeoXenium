#!/bin/bash



set -x -e
echo "copy data on `hostname`"

cd /home/zhongz2/ULIP

SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version3
SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version4
DST_DIR=/lscratch/$SLURM_JOB_ID

for subset in "train" "val"; do

mkdir -p ${DST_DIR}/${subset}

bash extract_tars.sh $SRC_DIR/${subset}_list.txt ${DST_DIR}/${subset}
cp $SRC_DIR/../X_pca_3.npy $DST_DIR/${subset}
cp $SRC_DIR/${subset}_items.pkl $DST_DIR/${subset}

done


exit;


SRC_DIR=/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version2
DST_DIR=/lscratch/$SLURM_JOB_ID
for f in `ls ${SRC_DIR}/patches*.tar.gz`; do 
tar -xf $f -C $DST_DIR/;
done
cp $SRC_DIR/X_pca_3.npy $DST_DIR/
cp $SRC_DIR/train_items.pkl $DST_DIR/


sleep 1;












