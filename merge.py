import numpy as np

sta = []

for k in range(1000):
    k_keys=np.load(f'outputs/100_40k_750_6,2,0,1,1_m4__unknot_wr/writhe{k}.npz',allow_pickle=True)
    for key in k_keys:
        temp=k_keys[key]
        for k in range(len(temp)):
            sta.append(temp[k])
        
np.savez_compressed('outputs/100k_40k_750_6,2,0,1,1_m4__unknot_wr_100_2200.npz',sta=np.array(sta))


#array0 = []
#array3 = []
#array4 = []
#
#for k in range(1000):
#    k_keys0=np.load(f'outputs/1200/100_40k_1200_6,2,0,1,1_m4__0/task{k}.npz',allow_pickle=True)
#    k_keys3=np.load(f'outputs/1200/100_40k_1200_6,2,0,1,1_m4__3_1/task{k}.npz',allow_pickle=True)
#    k_keys4=np.load(f'outputs/1200/100_40k_1200_6,2,0,1,1_m4__4_1/task{k}.npz',allow_pickle=True)
#    
#    for key in k_keys3:
#        array0.append(k_keys0[key])
#        array3.append(k_keys3[key])
#        array4.append(k_keys4[key])
        
#np.savez_compressed('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__0_xyz.npz',xyz=np.array(array0, dtype='object'))
#np.savez_compressed('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__3_1_xyz.npz',xyz=np.array(array3, dtype='object'))
#np.savez_compressed('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__4_1_xyz.npz',xyz=np.array(array4, dtype='object'))