

import glob,os,sys
import numpy as np
import pickle

# src_dir = sys.argv[1]
src_dir = '/data/zhongz2/Xenium_Prime_Mouse_Brain_Coronal_FF_outs/version3'

dirs = glob.glob(os.path.join(src_dir, 'rot*'))

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
    all_items = []
    for i in inds:
        d = dirs[i]
        file_list.append(os.path.join(d, 'patches.tar.gz'))
        with open(os.path.join(d, f'train_items_{num_patches}_{patch_size}.pkl'), 'rb') as fp:
            all_items.extend(pickle.load(fp))
    with open(os.path.join(src_dir, subset+'_list.txt'), 'w') as fp:
        fp.writelines('\n'.join(file_list))
        fp.write('\n')
    with open(os.path.join(src_dir, subset+'_items.pkl'), 'wb') as fp:
        pickle.dump(all_items, fp)




















