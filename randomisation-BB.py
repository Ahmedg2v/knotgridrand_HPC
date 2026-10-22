#!/usr/bin/env python
# coding: utf-8

# ## Packages

import numpy as np
import random
import sys

# ## Functions

def grid_path_3d(p0, p1):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    x, y, z = np.floor(p0).astype(int)
    x1, y1, z1 = np.floor(p1).astype(int)

    dx, dy, dz = p1 - p0
    dr = [dx,dy,dz]
    step = np.sign(dr).astype(int)

    inv_dr = []
    paradist = []
    for i in range(3):
        if dr[i] != 0:
            inv_dr.append(abs(1/dr[i]))
        else:
            inv_dr.append(np.inf)

    for j in range(3):
        if dr[j] == 0:
            paradist.append(np.inf)
        else:
            if step[j] > 0:
                dist = (np.floor(p0[j]) + 1 - p0[j])
            else:
                dist = (p0[j] - np.floor(p0[j]))

            paradist.append(dist * inv_dr[j])

    path = [(x, y, z)]

    nx = abs(x1 - x)
    ny = abs(y1 - y)
    nz = abs(z1 - z)

    while nx > 0 or ny > 0 or nz > 0:
        if paradist[0] <= paradist[1] and paradist[0] <= paradist[2]:
            paradist[0] += inv_dr[0]
            x += step[0]
            nx -= 1
        elif paradist[1] <= paradist[2]:
            paradist[1] += inv_dr[1]
            y += step[1]
            ny -= 1
        else:
            paradist[2] += inv_dr[2]
            z += step[2]
            nz -= 1

        path.append((x, y, z))

    return np.array(path, dtype=int)


# ### Generating Knots

def get_grid(x):
    knot = base_knots[x]
    grid_points = []
    for i in range(len(knot)):
        path = grid_path_3d(knot[i], knot[(i+1)%len(knot)])
        grid_points.extend(path)

    pts=np.array(grid_points,dtype=int)
    mask = np.any(np.diff(pts, axis=0) != 0, axis=1)
    # True to keep the first point
    mask = np.r_[True, mask]

    # result
    grid_points = pts[mask]
    grid_points = np.array(grid_points).astype(np.int64)

    return grid_points


def array_knot(gd, fwd):
    tlist=[]
    goto=list(fwd.keys())[0]
    for i in gd.keys():
        tlist.append(gd[fwd[goto]])
        goto=fwd[goto]

    grid_points=np.array(tlist)

    return grid_points


# ### Dictionary logic

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

def dictify_strand(x, key):
    grid_dict = {}
    forward_map = {}
    backward_map = {}

    for j in range(len(x)):
        grid_dict[j+key] = np.array([x[j][0],x[j][1],x[j][2]])
        forward_map[j+key] = (j+1)%len(x)+key
        backward_map[j+key] = (j-1)%len(x)+key

    grid_dict_check = { f"{v[0]},{v[1]},{v[2]}": k for k,v in grid_dict.items() }
    return grid_dict, forward_map, backward_map, grid_dict_check


def update_dict_add(x, index, move, grid_dict, forward_map, backward_map, grid_dict_check):
    '''add a new point after the point at index idx'''
    backward_map[forward_map[index]]=x
    forward_map[x]=forward_map[index]
    forward_map[index]=x
    backward_map[x]=index
    grid_dict[x]=move
    grid_dict_check[f"{move[0]},{move[1]},{move[2]}"]=x

    return grid_dict, forward_map, backward_map, grid_dict_check

def update_dict_remove(index, grid_dict, forward_map, backward_map, grid_dict_check):
    '''remove the point at index idx'''
    forward_map[backward_map[index]]=forward_map[index]
    backward_map[forward_map[index]]=backward_map[index]
    backward_map.pop(index)
    forward_map.pop(index)
    gpop = grid_dict[index]
    grid_dict_check.pop(f"{gpop[0]},{gpop[1]},{gpop[2]}")
    grid_dict.pop(index)

    return grid_dict, forward_map, backward_map, grid_dict_check

def update_dict_add_batch(index, str_start, str_end, str_dict, str_fwd, str_bwd, str_check, grid_dict, forward_map, backward_map, grid_dict_check):
    forward_map.update(str_fwd)
    backward_map.update(str_bwd)
    backward_map[forward_map[index]]=str_end
    forward_map[str_end]=forward_map[index]
    forward_map[index]=str_start
    backward_map[str_start]=index
    grid_dict.update(str_dict)
    grid_dict_check.update(str_check)

    return grid_dict, forward_map, backward_map, grid_dict_check


# ### Energy calculation

def energy_calc(gd, fwd, bwd, alpha, beta, gamma):
    dot_sum=0
    potential=0
    pts = np.array(list(gd.values()))

    x_mean = np.average(pts[:,0])
    y_mean = np.average(pts[:,1])
    z_mean = np.average(pts[:,2])

    for p in list(gd.keys()):
        bwdd = bwd[p]
        fwdd = fwd[p]

        gd0 = gd[bwdd]
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

def energy_calc_local(x, gd, alpha, beta, gamma):
    dot_sum=0
    potential=0
    pts = np.array(list(gd.values()))

    x_mean = np.average(pts[:,0])
    y_mean = np.average(pts[:,1])
    z_mean = np.average(pts[:,2])

    for p in range(5):  
        x0 = x[(p-1)%len(x)]
        x1 = x[p]
        x2 = x[(p+1)%len(x)]

        potential+=np.sqrt((x1[0]-x_mean)**2+
                           (x1[1]-y_mean)**2+
                           (x1[2]-z_mean)**2)

        dv0 = x2-x1
        dv1 = x1-x0

        dot_sum += np.dot(dv0, dv1)

    energy = alpha*len(x)+beta*(len(x)-dot_sum)+gamma*potential
    return energy


# ### Randomisation

# #### BFACF

def BFACF(gd, fwd, bwd, gdc, next_key, bx, by, bz, b_mat, energy, T, alpha, beta, gamma, theta):
    next_key = next_key+2
    idx = random.choice(list(gd.keys()))

    a1 = gd[idx]
    a2 = gd[fwd[idx]]
    a_1 = gd[bwd[idx]]
    a_2 = gd[bwd[bwd[idx]]]
    a3 = gd[fwd[fwd[idx]]]

    #b = ((a2-a1)+1)%2

    possible_moves = [[a1+bx,a2+bx],
                      [a1-bx,a2-bx],
                      [a1+by,a2+by],
                      [a1-by,a2-by],
                      [a1+bz,a2+bz],
                      [a1-bz,a2-bz]]

    die0 = random.choice(range(6))
    picked_move = possible_moves[die0]

    tup0 = f"{picked_move[0][0]},{picked_move[0][1]},{picked_move[0][2]}"
    tup1 = f"{picked_move[1][0]},{picked_move[1][1]},{picked_move[1][2]}"

    check0 = tup0 not in gdc
    check1 = tup1 not in gdc

    prev0 = f"{a_1[0]},{a_1[1]},{a_1[2]}" == tup0
    next1 = f"{a3[0]},{a3[1]},{a3[2]}" == tup1

    if check0 and check1: # no crossing
        delta_e = alpha*2 + beta*2 
        energy1 = energy+delta_e

        prob = np.exp(-theta*energy1/T)
        #flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0],1)

        flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0], 
                       np.where(delta_e < 0, random.choices([0,1],[prob,1-prob],k=1)[0], 1))

        if flip==1:
            gd, fwd, bwd, gdc = update_dict_add(next_key, idx, picked_move[0], gd, fwd, bwd, gdc)

            gd, fwd, bwd, gdc = update_dict_add(next_key+1, next_key, picked_move[1], gd, fwd, bwd, gdc)

            energy = energy1
        else:
            pass        

    elif prev0 and next1:
        delta_e = -alpha*2 - beta*2 
        energy1 = energy+delta_e

        prob = np.exp(-theta*energy1/T)
        #flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0],1)

        flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0], 
                       np.where(delta_e < 0, random.choices([0,1],[prob,1-prob],k=1)[0], 1))

        if flip==1:                
            gd, fwd, bwd, gdc = update_dict_remove(fwd[idx], gd, fwd, bwd, gdc)

            gd, fwd, bwd, gdc = update_dict_remove(idx, gd, fwd, bwd, gdc)

            energy = energy1
        else:
            pass

    elif prev0 and check1:
        local_pts_old = [a_2, a_1, a1, a2, a3]

        local_pts_new = [a_2, a_1, picked_move[1], a2, a3]  

        delta_e = energy_calc_local(local_pts_new, gd, alpha, beta, gamma)-energy_calc_local(local_pts_old, gd, alpha, beta, gamma)
        energy1 = energy+delta_e

        prob = np.exp(-theta*energy1/T)
        #flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0],1)

        flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0], 
                       np.where(delta_e < 0, random.choices([0,1],[prob,1-prob],k=1)[0], 1))

        if flip==1:
            gd, fwd, bwd, gdc = update_dict_add(next_key, bwd[idx], picked_move[1], gd, fwd, bwd, gdc)

            gd, fwd, bwd, gdc = update_dict_remove(idx, gd, fwd, bwd, gdc)

            energy = energy1
        else:
            pass

    elif next1 and check0:
        local_pts_old = [a_2, a_1, a1, a2, a3]

        local_pts_new = [a_2, a_1, picked_move[0], a2, a3]  

        delta_e = energy_calc_local(local_pts_new, gd, alpha, beta, gamma)-energy_calc_local(local_pts_old, gd, alpha, beta, gamma)
        energy1 = energy+delta_e

        prob = np.exp(-theta*energy1/T)
        #flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0],1)

        flip = np.where(delta_e > 0, random.choices([0,1],[1-prob,prob],k=1)[0], 
                       np.where(delta_e < 0, random.choices([0,1],[prob,1-prob],k=1)[0], 1))

        if flip==1:
            gd, fwd, bwd, gdc = update_dict_add(next_key, idx, picked_move[0], gd, fwd, bwd, gdc)

            gd, fwd, bwd, gdc = update_dict_remove(fwd[idx], gd, fwd, bwd, gdc)

            energy = energy1
        else:
            pass

    next_key+=1 

    return gd, fwd, bwd, gdc, next_key, energy


# #### Slide Move

def rng_slide(gd, fwd, bwd, gdc, next_key, bx, by, bz, b_mat, energy, T, alpha, beta, gamma, phi):

    next_key0 = next_key+5
    length=len(gd)
    die0 = random.sample(b_mat,2)
    axis = die0[0]*random.choice([1,-1])
    dist = random.choice(range(5,30)) 
    start_pt = random.choices(list(gd.keys()))[0]    #idx of the 1st point afterwhich the strand starts
    #die1 = random.choice(range(int(0.01*length),int(0.1*length)))
    die1 = random.choice(range(5,30))

    strand_idx = [] 
    strand_idx.append(fwd[start_pt])

    for u in range(die1):
        strand_idx.append(fwd[strand_idx[u]])

    end_pt=fwd[strand_idx[-1]]
    strand_pts = []

    for l in range(die1):
        strand_pts.append(gd[strand_idx[l]])

    strand_pts=np.array(strand_pts)
    ghost_check = [f"{v[0]},{v[1]},{v[2]}" for v in strand_pts]
    check_str = []         

    for d1 in range(1,dist):
        current=strand_pts+(d1*axis)
        comp=[f"{v[0]},{v[1]},{v[2]}" for v in current]
        check_str.append(any(p in gdc and p not in ghost_check for p in comp))

    new_strand0=strand_pts+(dist*axis)

    if random.choice([True, False]):
        axis2 = die0[1]*random.choice([1,-1])
        dist2 = random.choice(range(5,30))

        for d2 in range(1,dist2):
            current=new_strand0+(d2*axis2)
            comp=[f"{v[0]},{v[1]},{v[2]}" for v in current]
            check_str.append(any(p in gdc and p not in ghost_check for p in comp))

        new_strand1=new_strand0+(dist2*axis2)

        leg1 = grid_path_3d(gd[strand_idx[0]], new_strand0[0])
        leg1_= grid_path_3d(leg1[-1], new_strand1[0])[1:]
        leg2_= grid_path_3d(new_strand1[-1],new_strand0[-1])
        leg2 = grid_path_3d(leg2_[-1], gd[strand_idx[-1]])[1:]
        new_strand = np.vstack((leg1, leg1_,new_strand1[1:-1], leg2_,leg2))

    else:
        leg1 = grid_path_3d(gd[strand_idx[0]], new_strand0[0])
        leg2 = grid_path_3d(new_strand0[-1], gd[strand_idx[-1]])
        new_strand = np.vstack((leg1, new_strand0[1:-1], leg2))

    check_str = np.array(check_str)
    new_keys = [f"{p[0]},{p[1]},{p[2]}" for p in new_strand]

    if len(set(new_keys)) != len(new_keys) or any(k in gdc for k in new_keys[1:-1]) or any(check_str==True):
        pass

    else:
        delta_e=energy_calc_local(new_strand, gd, alpha, beta, gamma)-energy_calc_local(strand_pts, gd, alpha, beta, gamma)
        energy1=energy+delta_e
        prob = np.exp(-phi*energy1/T)
        #print(prob)

        if random.choices([0,1],[1-prob,prob],k=1)[0]==1:
            sd, sc, sb, sdc = dictify_strand(new_strand[:-1], next_key0)
            #print(new_strand)
            for n0 in range(len(strand_idx)-1):
                gd, fwd, bwd, gdc = update_dict_remove(strand_idx[n0], gd, fwd, bwd, gdc)

            gd, fwd, bwd, gdc = update_dict_add_batch(start_pt, next_key0, next_key0+len(sd)-1, sd, sc, sb, sdc, gd, fwd, bwd, gdc)

            worked=True
            next_key=next_key0+len(new_strand)+10

        else:
            pass

    return gd, fwd, bwd, gdc, next_key, energy


# #### Combined

def randomise(gd, fwd, bwd, gdc, t, next_key, T, alpha, beta, gamma, theta, phi):
    energy = energy_calc(gd, fwd, bwd, alpha, beta, gamma)

    bx = np.array([1,0,0])
    by = np.array([0,1,0])
    bz = np.array([0,0,1])
    b_mat = [bx,by,bz]

    for m in range(t):
        gd, fwd, bwd, gdc, next_key, energy = BFACF(gd, fwd, bwd, gdc, next_key, bx, by, bz, b_mat, energy, T, alpha, beta, gamma, theta)

        if m%4==0:
            gd, fwd, bwd, gdc, next_key, energy = rng_slide(gd, fwd, bwd, gdc, next_key, bx, by, bz, b_mat, energy, T, alpha, beta, gamma, phi)

    return gd, fwd, bwd, gdc, next_key


# ## Body
base_knots = np.load("base_knots_20.npz")
#grid_points = get_grid('4_1')

grid_points = np.vstack((grid_path_3d([-10,-10,0],[10,-10,0])[1:],grid_path_3d([10,-10,0],[10,10,0])[1:],grid_path_3d([10,10,0],[-10,10,0])[1:],grid_path_3d([-10,10,0],[-10,-10,0])[1:]))

def main(task_id):
    
    alpha, beta, gamma, theta, phi = 6, 2, 0, 1,1
    T=1200
    t=40000
    next_key = 10000
    
    base_seed = 123456 + task_id*1000

    knots=[]
    
    for k in range(100):
        seed = base_seed + k
        random.seed(seed)
        np.random.seed(seed)
        
        grid_dict, forward_map, backward_map, grid_dict_check = dictify(grid_points)
        
        grid_dict, forward_map, backward_map, grid_dict_check, next_key = randomise(grid_dict, forward_map, backward_map, grid_dict_check, t, next_key, T, alpha, beta, gamma, theta, phi)
        
        hknot2 = array_knot(grid_dict,forward_map)
        knots.append(hknot2)

    np.savez_compressed(f'outputs/1200/100_40k_1200_6,2,0,1,1_m4__0/task{task_id}.npz', *knots)
    
if __name__ == "__main__":
    task_id = int(sys.argv[1])
    main(task_id)