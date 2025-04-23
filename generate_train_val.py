

import glob,os,sys
import numpy as np
import pickle

# src_dir = sys.argv[1]
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version3'
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version4'
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version5'
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version6'
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version7_with_video'
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video'
# src_dir = '/mnt/gridftp/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version8_with_video'

dirs = os.listdir(src_dir)

train_ratio = 0.8
num_patches = 10000
patch_size = 448

np.random.shuffle(dirs)

train_count = int(np.floor(train_ratio * len(dirs)))
indices = {
    'train': np.arange(train_count),
    'val': np.arange(train_count, len(dirs))
}

for subset, inds in indices.items():
    file_list = []
    file_list_he = []
    file_list_video = []
    all_items = {}
    all_video_items = []
    for i in inds:
        d = dirs[i]
        file_list.append(os.path.join(d, 'patches_npy.tar.gz'))
        file_list_he.append(os.path.join(d, 'patches.tar.gz'))
        file_list_video.append(os.path.join(d, 'patches_video.tar.gz'))
        with open(os.path.join(src_dir, d, f'train_items_{num_patches}_{patch_size}.pkl'), 'rb') as fp:
            all_items.update(pickle.load(fp))
        with open(os.path.join(src_dir, d, f'train_items_{num_patches}_{patch_size}.csv'), 'r') as fp:
            all_video_items.extend(fp.readlines())
    with open(os.path.join(src_dir, subset+'_list_dapi.txt'), 'w') as fp:
        fp.writelines('\n'.join(file_list))
        fp.write('\n')
    with open(os.path.join(src_dir, subset+'_list_he.txt'), 'w') as fp:
        fp.writelines('\n'.join(file_list_he))
        fp.write('\n')
    with open(os.path.join(src_dir, subset+'_list_video.txt'), 'w') as fp:
        fp.writelines('\n'.join(file_list_video))
        fp.write('\n')
    with open(os.path.join(src_dir, subset+'_items.pkl'), 'wb') as fp:
        pickle.dump(all_items, fp)
    with open(os.path.join(src_dir, subset+'_items.csv'), 'w') as fp:
        fp.writelines(all_video_items)




















