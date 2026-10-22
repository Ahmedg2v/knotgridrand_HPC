import numpy as np
# writhe properties of the 3_1 and 4_1 knots, and the unknot, for comparison

wr_3=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__3_1_wr_100_2200.npz',allow_pickle=True)['sta'].flatten()
wr_4=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__4_1_wr_100_2200.npz',allow_pickle=True)['sta'].flatten()
wr_0=np.load('outputs/100k_40k_750_6,2,0,1,1_m4__unknot_wr_100_2200.npz',allow_pickle=True)['sta'].flatten()

vals0 = np.array([np.average(wr_0), np.std(wr_0), np.min(wr_0), np.max(wr_0)])
vals3 = np.array([np.average(wr_3), np.std(wr_3), np.min(wr_3), np.max(wr_3)])
vals4 = np.array([np.average(wr_4), np.std(wr_4), np.min(wr_4), np.max(wr_4)])

print(vals0,vals3,vals4)