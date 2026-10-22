import numpy as np

def dictify(x):
    grid_dict = {}
    forward_map = {}
    backward_map = {}

    for j in range(len(x)-1):
        grid_dict[j] = np.array([x[j][0],x[j][1],x[j][2]])
        forward_map[j] = (j+1)%(len(x)-1)
        backward_map[j] = (j-1)%(len(x)-1)

    grid_dict_check = { f"{v[0]},{v[1]},{v[2]}": k for k,v in grid_dict.items() }

    return grid_dict, forward_map, backward_map, grid_dict_check
    
def energy_calc(gd, fwd, bwd, alpha, beta, gamma):
    dot_sum=0
    potential=0
    pts = np.array(list(gd.values()))

    x_mean = np.average(pts[:,0])
    y_mean = np.average(pts[:,1])
    z_mean = np.average(pts[:,2])

    for p in list(gd.keys()):
        bwdb = bwd[p]
        fwdd = fwd[p]

        gd0 = gd[bwdb]
        gd1 = gd[p]
        gd2 = gd[fwdd]

        potential+=np.sqrt((gd1[0]-x_mean)**2+
                           (gd1[1]-y_mean)**2+
                           (gd1[2]-z_mean)**2)

        dv0 = gd2-gd1
        dv1 = gd1-gd0

        dot_sum += np.dot(dv0, dv1)

    energy = alpha*len(gd)+beta*(len(gd)-dot_sum)+gamma*potential
    return energy

arr_0=np.load('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__0_xyz.npz',allow_pickle=True)['xyz']
arr_3_1=np.load('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__3_1_xyz.npz',allow_pickle=True)['xyz']
arr_4_1=np.load('outputs/1200/100k_40k_1200_6,2,0,1,1_m4__4_1_xyz.npz',allow_pickle=True)['xyz']

coms0, lens0, ener0 = [],[],[]
coms3, lens3, ener3 = [],[],[]
coms4, lens4, ener4 = [],[],[]
for k in range(len(arr_0)):
    gd0, fwd0, bwd0, gdc0 = dictify(arr_0[k])
    coms0.append([np.average(arr_0[k][:,0]),np.average(arr_0[k][:,1]),np.average(arr_0[k][:,2])]) 
    lens0.append(len(arr_0[k]))
    ener0.append(energy_calc(gd0, fwd0, bwd0, 6, 2, 0))
    
    gd3, fwd3, bwd3, gdc3 = dictify(arr_3_1[k])
    coms3.append([np.average(arr_3_1[k][:,0]),np.average(arr_3_1[k][:,1]),np.average(arr_3_1[k][:,2])]) 
    lens3.append(len(arr_3_1[k]))
    ener3.append(energy_calc(gd3, fwd3, bwd3, 6, 2, 0))
    
    gd4, fwd4, bwd4, gdc4 = dictify(arr_4_1[k])
    coms4.append([np.average(arr_4_1[k][:,0]),np.average(arr_4_1[k][:,1]),np.average(arr_4_1[k][:,2])]) 
    lens4.append(len(arr_4_1[k]))
    ener4.append(energy_calc(gd4, fwd4, bwd4, 6, 2, 0))
    
coms0=np.array(coms0)
lens0=np.array(lens0)
ener0=np.array(ener0)

coms3=np.array(coms3)
lens3=np.array(lens3)
ener3=np.array(ener3)

coms4=np.array(coms4)
lens4=np.array(lens4)
ener4=np.array(ener4)

np.savez_compressed('stats_0,3_1,4_1.npz',coms0=coms0, lens0=lens0, ener0=ener0, coms3=coms3, lens3=lens3, ener3=ener3, coms4=coms4, lens4=lens4, ener4=ener4)
