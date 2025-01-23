

import sys,os,shutil,glob
import numpy as np
import pandas as pd


data_dir = '/data/zhongz2/data/pointllm/shapenet-55'

gscmd = 'gsutil -m cp "gs://sfr-ulip-code-release-research/shapenet-55/shapenet_pc/{}.npy" shapenet_pc/"{}.npy"'

files1 = glob.glob(os.path.join(data_dir, 'only_rgb_depth_images', '*'))
files2 = glob.glob(os.path.join(data_dir, 'rendered_images', '*'))
prefixes1 = [os.path.splitext(os.path.basename(f))[0] for f in files1]
prefixes2 = [os.path.basename(f) for f in files2]

for subset in ['train']:
    with open(os.path.join(data_dir, f'{subset}0.txt'), 'r') as fp:
        prefixes = fp.readlines()

    cmd_lines = []
    existed_prefixes = []
    for prefix0 in prefixes: 
        prefix = prefix0.strip().replace('.npy', '')
        count1 = len([v for v in prefixes1 if prefix == v[:len(prefix)]])
        exist2 = prefix in prefixes2
        if count1==60 and exist2:
            # os.system(gscmd.format(prefix, prefix))
            cmd_lines.append(gscmd.format(prefix, prefix)+'\n')
            existed_prefixes.append(prefix0)
            print(prefix0)
        
        if subset == 'train' and len(existed_prefixes) == 10000:
            break
    
    with open(os.path.join(data_dir, f'{subset}.txt'), 'w') as fp:
        fp.writelines(existed_prefixes)
    with open(os.path.join(data_dir, f'{subset}_cmd.txt'), 'w') as fp:
        fp.writelines(cmd_lines)












