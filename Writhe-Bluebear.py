#!/usr/bin/env python
# coding: utf-8


import numpy as np
import sys

def add_pts(arr, target):
    mids=(np.vstack((arr[1:],arr[0]))+arr)/2

    mask=np.full((len(mids),1),np.nan)
    mask[0:target-len(arr)]=1
    p=np.random.permutation(len(mask))
    mask=mask[p]
    mids=mids*mask

    tmp=np.zeros((len(mids)+len(arr),3))
    tmp[::2]=tmp[::2]+arr
    tmp[1::2]=tmp[1::2]+mids

    new_arr = tmp[np.isfinite(tmp).all(axis=1)]
    
    if (len(new_arr)<target):
        new_arr = add_pts(new_arr, target)
    else:
        new_arr = new_arr 
        
    return new_arr

def sta_writhe(wknot,width):
    with np.errstate(divide='ignore', invalid='ignore'):
        wtang = np.vstack((wknot[1:],wknot[0]))-wknot
        dots=[]
        mods=[]
        for kl in range(len(wknot)):
            vc = wknot[kl]-wknot
            cr = np.cross(wtang[kl], wtang)
            dot=[]
            for m in range(len(wknot)):
                dot.append(np.dot(cr[m], vc[m].T))

            dots.append(dot)
            mod = np.linalg.norm(vc,axis=1)**3
            mods.append(mod)

        dots=np.array(dots)
        mods=np.array(mods)
        frac=dots/mods
        frac[np.isnan(frac)]=0
        wr = np.sum(frac,axis=1).reshape(-1, 1)
    
        offset=30+width
        wr_padded = np.vstack((wr[-offset:],wr,wr[:offset]))
        sta=[]
        for i in range(len(wknot)):
            lim1, lim2 = int(offset+i-width/2), int(offset+i+width/2)
            sta.append(np.sum(wr_padded[lim1:lim2]))
        sta=np.array(sta).reshape(-1, 1)

    return sta

def main(task_id):
    knots = np.load(f'outputs/100_40k_750_6,2,0,1,1_m4__unknot/task{task_id}.npz',allow_pickle=True)
    target=2200
    writhes=[]

    for k in knots:
        writhes.append(sta_writhe(add_pts(knots[k],target),100))

    np.savez_compressed(f'outputs/100_40k_750_6,2,0,1,1_m4__unknot_wr/writhe{task_id}.npz', writhes=np.array(writhes))
    
if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)

