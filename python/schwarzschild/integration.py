import numpy as np

bt_RK4 = [[0, 1/2, 1/2, 1],[[0,0,0,0], [1/2,0,0,0], [0,1/2,0,0], [0,0,1,0]],[1/6, 1/3, 1/3, 1/6]]
bt_mpv = [[0,1/2],[[0,0],[1/2,0]],[0,1]]
bt_euler = [[0],[[0]],[1]]

# t, dt float, x numpy-array, dgl func(t,x)->dx/dt, bt butcher-tableau
def RK_step(t,x, dt, dgl, bt):
    RK_o = len(bt[0]) # Ordnung
    k = np.zeros((RK_o,*x.shape))
    for i in range(RK_o):
        sum = np.zeros(x.shape)
        for l in range(i):
            sum += bt[1][i][l]*k[l]
        k[i] = dgl(t+dt*bt[0][i],x+dt*sum)
    sum = np.zeros(x.shape)
    for i in range(RK_o):
        sum += bt[2][i]*k[i]
    return x+dt*sum

# r0,v0 2-D array, accF(x,y,vx,vy)=(ax,ay)
def get_geodesic(x0,y0,vx0,vy0,accF,dt = 0.1,N=10,bt=bt_RK4,bounds=[-100,-100,100,100],eps=1e-4,minNdiff=10):
    infos = np.array([[x0,y0,vx0,vy0]])
    taus = [0]

    def dgl(t,a):
        acc = accF(a[0],a[1],a[2],a[3])
        return np.array([a[2],a[3],acc[0],acc[1]])
    
    for i in range(N-1):
        taus.append(taus[-1]+dt)
        infos = np.append(infos, [RK_step(taus[-1],infos[-1], dt, dgl, bt)], axis=0)
        #infos.append(RK_step(taus[-1],infos[-1], dt, dgl, bt))
        if infos[-1][0] < bounds[0] or infos[-1][0] > bounds[2] or infos[-1][1] < bounds[1] or infos[-1][1] > bounds[3]:
            print(f'out of bounds: integration stopped')
            break
        if i > minNdiff:
            #dI = np.sum((infos-infos[-1])**2,axis=1)[:-minNdiff]/np.sum(infos**2,axis=1)[:-minNdiff]
            #if any(dI < eps**2):
            dr2 = distance2_point_to_segment(infos[0][:2], infos[-2][:2], infos[-1][:2])
            dv2 = distance2_point_to_segment(infos[0][2:], infos[-2][2:], infos[-1][2:])
            if dr2/np.sum(infos[-1][:2]**2) < eps and dv2/np.sum(infos[-1][2:]**2) < eps:
                print(f'recurring position & velocity: integration stopped')
                break

    return (np.array(taus),np.array(infos).T)

def distance2_point_to_segment(p, a, b):
    """
    Calculate the distance squared from point p to the line segment a-b.
    
    Args:
        p: Point (tuple) (x, y)
        a: Point (tuple) (x, y)
        b: Point (tuple) (x, y)
    
    Returns:
        float: The distance squared from p to the line segment a-b.
    """
    # Vector from a to b
    ab = (b[0] - a[0], b[1] - a[1])
    # Vector from a to p
    ap = (p[0] - a[0], p[1] - a[1])
    
    # Length of ab squared
    ab_length_sq = ab[0]**2 + ab[1]**2
    
    # If ab is zero, then a and b are the same point
    if ab_length_sq == 0:
        return ap[0]**2 + ap[1]**2
    
    # Parameter t
    t = (ap[0]*ab[0] + ap[1]*ab[1]) / ab_length_sq
    
    # Find the closest point on the segment
    if t < 0:
        closest = a
    elif t > 1:
        closest = b
    else:
        closest = (a[0] + t*ab[0], a[1] + t*ab[1])
    
    # Calculate the distance
    return (p[0]-closest[0])**2 + (p[1]-closest[1])**2