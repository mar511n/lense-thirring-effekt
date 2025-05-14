import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import brentq
import scipy.integrate as spint
import scipy.interpolate as spinter

def set_params_lense_thirring(mass, omega, radius=1):
    global S, M, R
    M = mass
    R = radius
    S = 2/5 * M * R**2 * omega

def check_break(ts,ys):
    global R
    r2 = np.sum(ys[1:4]**2)
    return r2 < R**2

def set_params_gaußian_surface(variance, A=1.0):
    global sigma,amplitude
    sigma = variance
    amplitude = A

def acc_gaußian_surface(tau, zs):
    global sigma, amplitude
    a = np.zeros(4)
    u,v,du,dv = zs[0],zs[1],zs[2],zs[3]
    a[:2] = zs[2:]
    nom = amplitude**2 * ((u**2 - sigma**2) * du**2 + (v**2 - sigma**2) * dv**2 + 2*u*v*du*dv)
    denom = sigma**2 * (2 * np.pi * sigma**6 * np.exp((u**2 + v**2)/(sigma**2)) + amplitude**2 * (u**2 + v**2))
    a[2] = u*nom/denom
    a[3] = v*nom/denom
    return a

def z_gaußian_surface(u,v):
    global sigma, amplitude
    return - amplitude * np.exp(-(u**2+v**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def metric_gaußian_surface(u,v):
    global sigma,amplitude
    t = amplitude**2 * np.exp(-(u**2 + v**2)/sigma**2)/(2*np.pi*sigma**6)
    return np.array([[1+t*u**2, t*u*v],[t*u*v, 1+t*v**2]])

def acc_lense_thirring(tau, zs):
    global S, M, R
    a = np.zeros(8)
    rs = zs[1:4]
    vs = zs[5:]
    r = np.sqrt(np.sum(rs*rs))
    a[:4] = zs[4:]
    a[5:] -= M*rs
    a[5:] += 2*np.cross(vs, [0, 0, S])
    a[5:] -= 6*S*rs[2]/(r*r) * np.cross(vs, rs)
    a[5:] /= r*r*r
    return a

def efield(rv):
    global S, M, R
    if len(rv.shape) != 2:
        raise Exception("rv should have shape (N,d) with d<3")
    if rv.shape[1] < 3:
        rv = np.pad(rv, ((0,0),(0,3-rv.shape[1])))
    elif rv.shape[1] > 3:
        raise Exception("rv should have shape (N,d) with d<3")
    r = np.linalg.norm(rv,axis=-1)[:,np.newaxis]
    idxs0 = np.argwhere(r==0)
    r[idxs0] = 1
    res = -M*rv/r**3
    res[idxs0] = rv*0
    return res
    r = np.linalg.norm(rv)
    if r == 0:
        return 0*rv
    return -M*rv/r**3

def bfield(rv):
    global S, M, R
    if len(rv.shape) != 2:
        raise Exception("rv should have shape (N,d) with d<3")
    if rv.shape[1] < 3:
        rv = np.pad(rv, ((0,0),(0,3-rv.shape[1])))
    elif rv.shape[1] > 3:
        raise Exception("rv should have shape (N,d) with d<3")
    Sv = np.array([0,0,S])
    r = np.linalg.norm(rv,axis=-1)[:,np.newaxis]
    idxs0 = np.argwhere(r==0)
    r[idxs0] = 1
    dot = np.sum(rv*Sv, axis=1)[:,np.newaxis]
    res = 2*(Sv-3*dot/r**2 * rv)/r**3
    res[idxs0] = rv*0
    return res
    r = np.linalg.norm(rv)
    if r == 0:
        return 0*rv
    Sv = np.array([0,0,S])
    return (Sv-3*np.dot(Sv,rv)/r**2 * rv)/r**3

class ParametricCurve:
    """
    Class to handle operations on a parametrized curve
    """

    def __init__(self, ts, xs=None, xf=None, init_nat_param=True):
        self.ts = np.array(ts)
        self.tmax = np.max(self.ts)
        self.tmin = np.min(self.ts)
        if xf is not None:
            self.xs = xf(self.ts)
        elif xs is not None:
            self.xs = np.array(xs)
        else:
            raise Exception("either xs or xf has to be given")
        if init_nat_param:
            self.init_natural_parametrization()
    
    def init_natural_parametrization(self):
        l = np.append([0], np.sqrt(np.sum((self.xs[1:]-self.xs[:-1])**2,axis=1)),axis=0)
        l = np.cumsum(l)
        self.length = np.max(l)
        self.l_of_t = spinter.interp1d(self.ts, l, fill_value=(
            l[0], l[-1]), bounds_error=False, axis=0)
        self.t_of_l = spinter.interp1d(l, self.ts, fill_value=(
            self.ts[0], self.ts[-1]), bounds_error=False, axis=0)
        self.x_of_l = spinter.interp1d(l, self.xs, fill_value=(
            self.xs[0], self.xs[-1]), bounds_error=False, axis=0)
        #self.x_of_t = spinter.interp1d(self.ts, self.xs, fill_value=(
        #    self.xs[0], self.xs[-1]), bounds_error=False, axis=0)
        vs = np.gradient(l, self.ts)
        vs = np.where(np.isnan(vs), 0, vs)
        self.vmax = np.max(vs)
        self.vmin = np.min(vs)
        self.v_of_l = spinter.interp1d(l, vs, fill_value=(
            vs[0], vs[-1]), bounds_error=False, axis=0)
    
    def get_section_by_l(self, lstart, lend, seg_size=None):
        """
        returns an array of equally spaced l values from lstart to lend with approx segment size of seg_size
        """
        if seg_size is None:
            seg_size = self.length/100
        return np.linspace(lstart, lend, 1+int(np.round((lend-lstart)/seg_size)))
    
    def get_section_by_t(self, tstart, tend, seg_size=None):
        """
        returns an array of equally spaced l values corresponding to values from tstart to tend with approx segment size of seg_size
        """
        return self.get_section_by_l(self.l_of_t(tstart), self.l_of_t(tend), seg_size=seg_size)
    

def get_geodesic2(accF,z0,system_timescale,tmax=np.Inf,t_fw=True,tol=1e-7,cputmax=1.0,boundary=np.array([[-10,-10,-10],[10,10,10]]),recurring_tol=1e-1):
    tm = tmax if t_fw else -tmax
    rkdp = spint.RK45(accF, 0.0, z0, tm , first_step=system_timescale*tol*1e-2, max_step=system_timescale*1e2, rtol=tol, atol=tol*1e-2/system_timescale)
    ts = [0.0]
    rs = [rkdp.y]
    check_recurring= False
    trun0 = time.time()
    while time.time()-trun0 < cputmax:
        rkdp.step()
        if rkdp.h_abs > system_timescale*1e2:
            rkdp.status = f"h_abs too large: {rkdp.h_abs}"
        if any(np.logical_or(rkdp.y < boundary[0], rkdp.y > boundary[1])):
            rkdp.status = "out of bounds"
        dis = np.sum((np.array(rs[0]) - rkdp.y)**2)
        if check_recurring and dis < recurring_tol**2:
            rkdp.status = "returned to start point"
        if dis > recurring_tol**2:
            check_recurring = True
        ts.append(rkdp.t)
        rs.append(rkdp.y)
        if rkdp.status != "running":
            break
    ts,rs = np.array(ts), np.array(rs)
    sorter = np.argsort(ts)
    return ts[sorter], rs[sorter], rkdp.status

def get_geodesic(tau_max, z0, tol=1e-6, accF=acc_lense_thirring, check_break=check_break):
    rkdp = spint.RK45(accF, 0.0, z0, tau_max,
                      first_step=tol*1e-2, rtol=tol, atol=tol*1e-2)
    taus = [0.0]
    zs = [rkdp.y]
    while rkdp.y[0] < tau_max and rkdp.t < tau_max:
        rkdp.step()
        taus.append(rkdp.t)
        zs.append(rkdp.y)
        if not check_break is None:
            if check_break(rkdp.t, rkdp.y):
                break
        if rkdp.status != "running":
            break
    return np.array(taus),np.array(zs)

def generate_samples_scaled(g, a, b, N, rand=True):
    # Compute the normalizing constant
    integral, _ = quad(g, a, b)
    g_normalized = lambda x: g(x) / integral
    
    # Define the CDF
    def cdf(x):
        integral, _ = quad(g_normalized, a, x)
        return integral
    
    # Define the inverse CDF using root finding
    def inverse_cdf(u):
        def func(y):
            return cdf(y) - u
        result, _ = brentq(func, a, b, full_output=True)
        return result
    
    # Generate uniform samples and apply inverse CDF
    uniform_samples = np.linspace(1e-10,1-1e-10,N)
    if rand:
        uniform_samples = np.random.rand(N)
    samples = np.array([inverse_cdf(u) for u in uniform_samples])
    return np.sort(samples)

def generate_3d_grid_lines(start_point, end_point, num_lines, subdivisions, g):
    # Unpack the start and end points
    x0, y0, z0 = start_point
    x1, y1, z1 = end_point

    # Unpack the number of lines in each direction
    nx, ny, nz = num_lines

    # Generate the grid lines in each direction
    x_lines = np.linspace(x0, x1, nx)
    y_lines = np.linspace(y0, y1, ny)
    z_lines = np.linspace(z0, z1, nz)

    # Generate the points on each line with the specified number of subdivisions
    lines = []

    # Lines parallel to the yz-plane
    z_points = generate_samples_scaled(g, z0, z1, subdivisions, rand=False)
    for x in x_lines:
        for y in y_lines:
            line = np.column_stack(
                (np.full(subdivisions, x), np.full(subdivisions, y), z_points))
            lines.append(line.tolist())

    # Lines parallel to the xz-plane
    y_points = generate_samples_scaled(g, y0, y1, subdivisions, rand=False)
    for x in x_lines:
        for z in z_lines:
            line = np.column_stack(
                (np.full(subdivisions, x), y_points, np.full(subdivisions, z)))
            lines.append(line.tolist())

    # Lines parallel to the xy-plane
    x_points = generate_samples_scaled(g, x0, x1, subdivisions, rand=False)
    for y in y_lines:
        for z in z_lines:
            line = np.column_stack(
                (x_points, np.full(subdivisions, y), np.full(subdivisions, z)))
            lines.append(line.tolist())

    return np.array(lines)