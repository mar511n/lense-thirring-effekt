import numpy as np

# x is a (3,...) array with coordinates in sa, which will be converted to sb
# sa/sb can be: "cartesian", "cylindrical", "spherical"
def conv_coords(x,sa,sb):
    if sa == sb:
        return x
    y = np.zeros_like(x)
    if sa == "cartesian":
        if sb == "cylindrical":
            y[0] = np.sqrt(x[0]**2+x[1]**2)
            y[1] = np.arctan2(x[1],x[0])
            y[2] = x[2]
        else:
            y[0] = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
            y[1] = np.arctan2((np.sqrt(x[0]**2+x[1]**2)),x[2])
            y[2] = np.arctan2(x[1],x[0])
    elif sa == "cylindrical":
        if sb == "cartesian":
            y[0] = x[0]*np.cos(x[1])
            y[1] = x[0]*np.sin(x[1])
            y[2] = x[2]
        else:
            y[0] = np.sqrt(x[0]**2+x[2]**2)
            y[1] = np.arctan2(x[0],x[2])
            y[2] = np.arctan2(x[1])
    elif sa == "spherical":
        if sb == "cartesian":
            y[0] = x[0]*np.sin(x[1])*np.cos(x[2])
            y[1] = x[0]*np.sin(x[1])*np.sin(x[2])
            y[2] = x[0]*np.cos(x[1])
        else:
            y[0] = x[0]*np.sin(x[1])
            y[1] = x[2]
            y[2] = x[0]*np.cos(x[1])
    return y

# x,v are (3,...) array with coordinates/vector components in sa, which will be converted to sb
# sa/sb can be: "cartesian", "cylindrical", "spherical"
def conv_vec(x,v,sa,sb):
    if sa == sb:
        return v
    u = np.zeros_like(x)
    if sa == "cartesian":
        if sb == "cylindrical":
            u[0] = (x[0]*v[0]+x[1]*v[1])/np.sqrt(x[0]**2+x[1]**2)
            u[1] = (-x[1]*v[0]+x[0]*v[1])/np.sqrt(x[0]**2+x[1]**2)
            u[2] = v[2]
        else:
            u[0] = (x[0]*v[0]+x[1]*v[1]+x[2]*v[2])/np.sqrt(x[0]**2+x[1]**2+x[2]**2)
            u[1] = (x[0]*x[2]/np.sqrt(x[0]**2+x[1]**2)*v[0]+x[1]*x[2]/np.sqrt(x[0]**2+x[1]**2)*v[1]-np.sqrt(x[0]**2+x[1]**2)*v[2])/np.sqrt(x[0]**2+x[1]**2+x[2]**2)
            u[2] = (-x[1]*v[0]+x[0]*v[1])/np.sqrt(x[0]**2+x[1]**2)
    elif sa == "cylindrical":
        if sb == "cartesian":
            u[0] = np.cos(x[1])*v[0] - np.sin(x[1])*v[1]
            u[1] = np.sin(x[1])*v[0] + np.cos(x[1])*v[1]
            u[2] = v[2]
        else:
            u[0] = (x[0]*v[0]+x[2]*v[2])/np.sqrt(x[0]**2+x[2]**2)
            u[1] = (x[2]*v[0]-x[0]*v[2])/np.sqrt(x[0]**2+x[2]**2)
            u[2] = v[1]
    elif sa == "spherical":
        if sb == "cartesian":
            u[0] = np.sin(x[1])*np.cos(x[2])*v[0] + np.cos(x[1])*np.cos(x[2])*v[1] - np.sin(x[2])*v[2]
            u[1] = np.sin(x[1])*np.sin(x[2])*v[0] + np.cos(x[1])*np.sin(x[2])*v[1] + np.cos(x[2])*v[2]
            u[2] = np.cos(x[1])*v[0] - np.sin(x[1])*v[1]
        else:
            u[0] = np.sin(x[1])*v[0] + np.cos(x[1])*v[1]
            u[1] = v[2]
            u[2] = np.cos(x[1])*v[0] - np.sin(x[1])*v[1]
    return u