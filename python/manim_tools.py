from manimlib import *
import numpy as np
import scipy.integrate as spint
from scipy.spatial.transform import Rotation
import time
import lense_thirring_tools as ltt

class Arrow3D(Surface):
    """
    A 3D arrow object
    """
    def __init__(self,
                start = np.array([0,0,0]),
                end = np.array([0,0,1]),
                color = GREY,
                shading = (0.3, 0.2, 0.4),
                depth_test = True,
                tip_width_ratio = 0.3,
                tip_length = 0.1,
                shaft_width = 0.015,
                resolution = (41, 101),
                prefered_creation_axis = 1,
                epsilon = 1e-4,
                **kwargs):
        self.start = np.array(start)
        self.end = np.array(end)
        self.length = np.linalg.norm(self.end-self.start)
        self.tip_width_ratio = tip_width_ratio
        self.tip_length = tip_length
        self.shaft_width = shaft_width
        super().__init__(color, shading, depth_test, (0,TAU), (0,1.0), resolution, prefered_creation_axis, epsilon, **kwargs)
        self.move_to((self.start+self.end)/2)
        # rotate the arrow to point from start to end (currently it points along the z-axis)
        target = self.end-self.start
        self.rotate(np.arccos(target[2]/self.length), axis=(1,0,0))
        self.rotate(np.arctan2(target[1], target[0])+np.pi/2, axis=(0,0,1))

    def uv_func(self, u: float, v: float) -> np.ndarray:
        if v < 0.25:
            r = v*4*self.shaft_width
            return np.array([r*np.cos(u), r*np.sin(u), 0])
        elif v < 0.5:
            r = self.shaft_width
            return np.array([r*np.cos(u), r*np.sin(u), (v-0.25)*4*(self.length-self.tip_length)])
        elif v < 0.75:
            r = self.shaft_width+(v-0.5)*4*self.tip_width_ratio*self.tip_length
            return np.array([r*np.cos(u), r*np.sin(u), self.length-self.tip_length])
        r = (self.shaft_width+self.tip_width_ratio*self.tip_length)*(1.0-v)*4
        return np.array([r*np.cos(u), r*np.sin(u), self.length-self.tip_length*4*(1.0-v)])
        

class LineAnim(VGroup):
    """
    display and animate multiple lines based on a set of lines and timestamps
    ts.shape = (timesteps, )
    lines.shape = (line_nums, timesteps, subdivisions, 3)
    color_values.shape = lines.shape
    """
    def __init__(self, ts, lines, color_values=None, cmap="viridis", repeat=True, depth_test=True, basecolor=None, **kwargs):
        """
        TODO: color_values has not been tested yet...
        """
        self.line_nums = lines.shape[0]
        self.timesteps = len(ts)
        self.subdivisions = lines.shape[2]
        self.ts = ts
        self.lines = lines
        self.currentT = 0.0
        self.currentI = 0
        self.t_max = np.max(ts)
        self.repeat = repeat
        vmobjs = [VMobject(color=basecolor) for li in range(self.line_nums)]
        super().__init__(vmobjs, **kwargs)
        self.cmap = get_color_map(cmap)
        if color_values is not None:
            color_values -= np.min(color_values)
            color_values /= np.max(color_values)
            self.colors = self.cmap(color_values)
        else:
            self.colors = None

        if depth_test:
            self.apply_depth_test(recurse=True)
    
    def reset_state(self):
        self.stopUpdating()
        self.currentT = 0.0
        self.currentI = 0
        self.updateVMobjs(0,force=True)

    def startUpdating(self, timeScaleF=1.0, call=True):
        self.currentT = 0.0
        self.currentI = 0
        def updater(obj,dt):
            nonlocal timeScaleF
            obj.currentT += dt*timeScaleF
            if obj.currentT > obj.t_max:
                if obj.repeat:
                    obj.currentT -= obj.t_max
                    obj.currentI = 0
                else:
                    obj.currentT = obj.t_max
            obj.updateVMobjs(obj.currentT)
        self.add_updater(updater,call=call)
    
    def stopUpdating(self):
        self.clear_updaters()
        
    def updateVMobjs(self, t, force=False):
        changed = False
        while self.currentI+1 < len(self.ts) and self.ts[self.currentI+1] < t:
            self.currentI += 1
            changed = True
        
        if changed or force:
            for li in range(self.line_nums):
                self[li].set_points_smoothly(self.lines[li,self.currentI,:,:])
                if self.colors is not None:
                    self[li].set_stroke(color=self.colors[li,self.currentI])

class CurveDrawer(VMobject):
    """
    Draws multiple ParametricCurve objects and arrows
    """
    def __init__(self, pcs, tip_width_ratio=4, arrow_base_length=0.4, seg_base_size=0.1, col_func=lambda x: x, cmap="viridis", arrow_res = 4, vmin=None, vmax=None, fixed_color=None, randomize_t0s=True, **kwargs):
        super().__init__(**kwargs)
        self.apply_depth_test()
        self.stroke_base_width = self.stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.arrow_res = arrow_res
        self.fixed_color = fixed_color
        self.col_func = col_func
        self.col_map = get_color_map(cmap)
        self.randomize_t0s = randomize_t0s
        #self.arrow_base_length_factor = arrow_base_length_factor
        #self.seg_base_size_factor = seg_base_size_factor
        self.arrow_base_length = arrow_base_length
        self.seg_base_size = seg_base_size
        self.set_pcs(pcs,vmin=vmin, vmax=vmax)
        
    
    def set_pcs(self,pcs,vmin=None, vmax=None):
        self.pcs = pcs
        self.dt0s = np.zeros(len(self.pcs))
        if self.randomize_t0s:
            self.dt0s = np.random.random(len(self.pcs))
        self.lengths = np.array([pc.length for pc in pcs])
        mL = np.mean(self.lengths)
        #self.arrow_base_length = mL*self.arrow_base_length_factor
        #self.seg_base_size = mL*self.seg_base_size_factor
        if vmin is None:
            self.vmin = np.min([pc.vmin for pc in pcs])
        else:
            self.vmin = vmin
        if vmax is None:
            self.vmax = np.max([pc.vmax for pc in pcs])
        else:
            self.vmax = vmax

    
    def update_graphics(self, arrow_tss=None, arrow_length=None, seg_size=None):
        if arrow_length is None:
            arrow_length = self.arrow_base_length
        if seg_size is None:
            seg_size = self.seg_base_size
        if arrow_tss is None:
            arrow_tss = [[] for pc in self.pcs]
        if len(arrow_tss) != len(self.pcs):
            arrow_tss = np.array([arrow_tss+dt0 for dt0 in self.dt0s])
            arrow_tss = np.sort(arrow_tss % 1.0, axis=1)
        self.clear_points()
        sws = np.array([])
        svs = np.array([])

        for i,pc in enumerate(self.pcs):
            bigW_ls = []
            eval_ls = np.array([0])
            for at in arrow_tss[i]:
                at = (at % 1.0) * (pc.tmax-pc.tmin) + pc.tmin
                al = pc.l_of_t(at)
                if al-arrow_length/2 < eval_ls[-1] or al+arrow_length/2 < eval_ls[-1] or al+arrow_length/2 > pc.length:
                    continue
                eval_ls = np.append(eval_ls, pc.get_section_by_l(eval_ls[-1], al-arrow_length/2, seg_size)[1:],axis=0)
                bigW_ls.append(len(eval_ls))
                eval_ls = np.append(eval_ls, np.linspace(eval_ls[-1]+arrow_length/10, al+arrow_length/2,self.arrow_res),axis=0)
            eval_ls = np.append(eval_ls, pc.get_section_by_l(eval_ls[-1], pc.length, seg_size)[1:],axis=0)
            anchors = pc.x_of_l(eval_ls)
            points = np.empty((2 * len(anchors) - 1, 3))#resize_array(np.empty_like(anchors), 2 * len(anchors) - 1)
            points[0::2] = anchors
            points[1::2] = 0.5 * (anchors[:-1] + anchors[1:])
            self.add_subpath(points)
            pl = 1
            if i == len(self.pcs)-1:
                pl = 0
            sw = np.ones(len(points)+pl)*self.stroke_base_width
            for bi in bigW_ls:
                sw[2*bi:2*(bi+self.arrow_res):2] = np.linspace(self.tip_width_ratio, 1, self.arrow_res)*self.stroke_base_width
                sw[2*bi+1:2*(bi+self.arrow_res)+1:2] = sw[2*bi:2*(bi+self.arrow_res):2]
            #sw[2*np.array(bigW_ls, dtype=int)] = np.ones(len(bigW_ls))*self.stroke_base_width*self.tip_width_ratio
            sws = np.append(sws, sw, axis=0)
            svs = np.append(svs, pc.v_of_l(eval_ls),axis=0)
        svs = self.col_func((svs-self.vmin)/(self.vmax-self.vmin))
        self.get_stroke_widths()[:] = sws
        if self.fixed_color is None:
            cols = [rgba_to_color(cv) for cv in self.col_map(svs)]
            self.set_color(cols)
        else:
            self.set_color(self.fixed_color)

def get_grid_surface(uv_func, grid_size=(6,6), grid_resolution=(101,101), grid_col=WHITE, u_range=(0,1), v_range=(0,1), grid_off=1e-2, **kwargs):
        rs = [np.array([np.linspace(v_range[0],v_range[1],grid_resolution[1]),np.ones(grid_resolution[1])*u,np.linspace(v_range[0],v_range[1],grid_resolution[1])]) for u in np.linspace(u_range[0],u_range[1],grid_size[0]+1)]
        rs.extend([np.array([np.linspace(u_range[0],u_range[1],grid_resolution[0]),np.linspace(u_range[0],u_range[1],grid_resolution[0]),np.ones(grid_resolution[0])*v]) for v in np.linspace(v_range[0],v_range[1],grid_size[1]+1)])
        pcs = [ltt.ParametricCurve(ps[0], np.array(uv_func(ps[1],ps[2])).T+np.array([0,0,grid_off])) for ps in rs]
        ps = ParametricSurface(uv_func, u_range, v_range, **kwargs)
        ps.apply_depth_test()
        cd = CurveDrawer(pcs, fixed_color=grid_col, stroke_width=2)
        cd.update_graphics()
        return cd, ps

class StreamLines(CurveDrawer):
    """
    Generates the ParametricCurves to draw and animates the Arrows
    """
    def __init__(self, fieldf=lambda t, x: np.array([x[1]**2,1,x[0]]), arrow_num=2, startPoints=None, startBox=[np.linspace(-1.5,1.5,3) for i in range(3)], boundary=None, system_timescale=1, tol=1e-7, recurring_tol=1e-1, trunmax=0.5, tip_width_ratio=4, arrow_base_length=0.4, seg_base_size=0.1, col_func=lambda x: x, cmap="viridis", arrow_res=4, **kwargs):
        if startPoints is None:
            startPoints = np.array([[x,y,z] for x in startBox[0] for y in startBox[1] for z in startBox[2]])
        if boundary is None:
            boundary = [np.min(startPoints,axis=0),np.max(startPoints,axis=0)]
            ml = np.max(boundary[1]-boundary[0])
            boundary[0] -= np.ones((3,))*ml/4
            boundary[1] += np.ones((3,))*ml/4
        self.arrow_ts = np.array([1/(2*arrow_num)+i/arrow_num for i in range(arrow_num)])
        self.startPoints = startPoints
        self.boundary = np.array(boundary)
        self.field = fieldf
        self.tol = tol
        self.system_timescale = system_timescale
        self.recurring_tol = recurring_tol
        self.trunmax = trunmax
        self.Dt = 0.0
        pcs = []
        print(f"calculating parametric curves for {len(startPoints)} start points")
        for sp in startPoints:
            t_fw,rs_fw = self.calculate_sl(sp, True)
            t_bw,rs_bw = self.calculate_sl(sp, False)
            ts = np.empty((len(t_fw)+len(t_bw)-1))
            ts[:len(t_bw)] = t_bw[::-1]
            ts[len(t_bw):] = t_fw[1:]
            rs = np.empty((len(rs_fw)+len(rs_bw)-1, 3))
            rs[:len(rs_bw)] = rs_bw[::-1]
            rs[len(rs_bw):] = rs_fw[1:]
            pcs.append(ltt.ParametricCurve(ts+np.min(ts), rs))
            print(">",end="")
        print("\nparametric curves initialized")
        super().__init__(pcs, tip_width_ratio, arrow_base_length, seg_base_size, col_func, cmap, arrow_res, **kwargs)
        self.update_graphics()

    def startUpdating(self, timeScaleF=1.0):
        self.Dt = 0.0
        def updater(obj,dt):
            nonlocal timeScaleF
            obj.Dt += dt*timeScaleF
            obj.update_graphics()
        self.add_updater(updater,call=True)
    
    def stopUpdating(self):
        self.clear_updaters()

    def update_graphics(self, arrow_length=None, seg_size=None):
        return super().update_graphics(self.arrow_ts+self.Dt, arrow_length, seg_size)

    def calculate_sl(self, pnt, t_fw=True):
        tm = np.Inf if t_fw else -np.Inf
        rkdp = spint.RK45(self.field, 0.0, pnt, tm , first_step=self.system_timescale*self.tol*1e-2, max_step=self.system_timescale*1e2, rtol=self.tol, atol=self.tol*1e-2/self.system_timescale)
        ts = [0.0]
        rs = [rkdp.y]
        check_recurring= False
        trun0 = time.time()
        while time.time()-trun0 < self.trunmax:
            rkdp.step()
            if rkdp.h_abs > self.system_timescale*1e2:
                rkdp.status = f"h_abs too large: {rkdp.h_abs}"
            if any(np.logical_or(rkdp.y < self.boundary[0], rkdp.y > self.boundary[1])):
                rkdp.status = "out of bounds"
            dis = np.sum((np.array(rs[0]) - rkdp.y)**2)
            if check_recurring and dis < self.recurring_tol**2:
                rkdp.status = "returned to start point"
            if dis > self.recurring_tol**2:
                check_recurring = True
            ts.append(rkdp.t)
            rs.append(rkdp.y)
            if rkdp.status != "running":
                break
        return np.array(ts), np.array(rs)

class CircularCamRotater:
    def __init__(self, frame, omega, rv0 = np.array([1.0,-1.0,1.1]), rotvec = np.array([1.0,-1.0,0.7]), upVec = np.array([0,0,1])):
        self.rv0 = rv0/np.linalg.norm(rv0)
        self.rotvec = rotvec/np.linalg.norm(rotvec)
        self.upVec = upVec/np.linalg.norm(upVec)
        self.omega = omega
        self.frame = frame
        self.t = 0
        self.update(frame,0)
    
    def startUpdating(self):
        self.t = 0
        self.frame.add_updater(self.update)
    
    def stopUpdating(self):
        self.t = 0
        self.frame.remove_updater(self.update)

    def get_Rotation(self):
        rot = Rotation.from_rotvec(self.rotvec*self.omega*self.t)
        z = rot.apply(self.rv0)
        x = np.cross(self.upVec, z)
        x /= np.linalg.norm(x)
        y = np.cross(z,x)
        return Rotation.from_matrix(np.array([x,y,z])).inv()

    def update(self, frame, dt):
        self.t += dt
        frame.set_orientation(self.get_Rotation())