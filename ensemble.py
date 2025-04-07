import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    npz_data = np.load('/home/yukaichen/code/g-CTR-GCN-main/CTR-GCN-main/data/ntu/NTU60_CS.npz')
    label = np.where(npz_data['y_test'] > 0)[1]


    with open(os.path.join('', ''), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join('', ''), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(os.path.join('', ''), 'rb') as r3:
        r3 = list(pickle.load(r3).items())
    
    with open(os.path.join('', ''), 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    right_num = total_num = right_num_5 = 0

    acc=[]        
    alpha1=0.6
    alpha2=0.6
    alpha3=0.4
    alpha4=0.4

    for i in tqdm(range(len(label))):
        l = label[i]

        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]

        r = r11 * alpha1 + r22 * alpha2 + r33 * alpha3 + r44 * alpha4 
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1  


    acc1 = right_num / total_num
    acc5 = right_num_5 / total_num
    acc.append(acc1)

    a=max(acc) 
    print('Top1 Acc: {:.4f}%'.format(acc1 * 100),format(acc5 * 100)) 
