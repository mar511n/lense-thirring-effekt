from manimlib import *
import numpy as np
import scipy.integrate as spint
import scipy.interpolate as spinter
from scipy.spatial.transform import Rotation

class Trajectories(Group):
    """
    display and animate the trajectories of particles based on a set of points and timestamps
    """
    def __init__(self, tmax, tnum, ts, ps, cmap="viridis", cfunc=None, repeat=True, depth_test=True, ball_r=0.1, ball_col=GREY, **kwargs):
        self.num = len(ts)
        self.trajs = VGroup([VMobject(**kwargs) for i in range(self.num)])
        self.balls = Group([Sphere(resolution=(21,11),radius=ball_r,color=ball_col) for i in range(self.num)]) 
        super().__init__([self.trajs, self.balls], **kwargs)
        self.repeat = repeat
        self.currentT = 0.0
        self.currentI = 0
        self.t_max = tmax
        self.ts = np.linspace(0,tmax,tnum)
        self.ps = np.empty((self.num,tnum,3))
        for i in range(self.num):
            self.ps[i] = spinter.interp1d(ts[i], ps[i].T, fill_value=(ps[i][0],ps[i][-1]), bounds_error=False)(self.ts).T
        self.cmap = get_color_map(cmap)
        cvs = []
        for i in range(self.num):
            vs = (self.ps[i][1:]-self.ps[i][:-1])
            vs = np.sqrt(np.sum(vs**2, axis=1))
            vs /= (self.ts[1:]-self.ts[:-1])
            vs = np.append([vs[0]],vs,axis=0)
            cvs.append(vs)
        cvs = np.array(cvs)
        cvs -= np.min(cvs)
        cvs /= np.max(cvs)
        if cfunc:
            cvs = cfunc(cvs)
        self.cols = [[rgba_to_color(col) for col in self.cmap(cvs[i])] for i in range(self.num)]
        self.pf = spinter.interp1d(self.ts, self.ps, fill_value=(self.ps[:,0],self.ps[:,-1]), bounds_error=False, axis=1)
        if depth_test:
            self.apply_depth_test(recurse=True)
    
    def startUpdating(self, timeScaleF=1.0):
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
        self.add_updater(updater,call=True)
    
    def stopUpdating(self):
        self.clear_updaters()

    def updateVMobjs(self, t):
        while self.currentI+1 < len(self.ts) and self.ts[self.currentI+1] < t:
            self.currentI += 1
        
        cp = self.pf(t)
        for i in range(self.num):
            ps = np.append(self.ps[i,:self.currentI+1], [cp[i]], axis=0)
            cols = np.append(self.cols[i][:self.currentI+1], [self.cols[i][self.currentI]], axis=0)
        
            self.trajs[i].set_points_smoothly(ps)
            self.trajs[i].set_stroke(color=cols)
            self.balls[i].move_to(cp[i])

class LineAnim(VGroup):
    """
    display and animate multiple lines based on a set of lines and timestamps
    ts.shape = (timesteps, )
    lines.shape = (line_nums, timesteps, subdivisions, 3)
    color_values.shape = lines.shape
    """
    def __init__(self, ts, lines, color_values=None, cmap="viridis", repeat=True, depth_test=True, **kwargs):
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
        vmobjs = [VMobject() for li in range(self.line_nums)]
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
    
    def startUpdating(self, timeScaleF=1.0):
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
        self.add_updater(updater,call=True)
    
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

class ParametricCurve:
    """
    Class to handle operations on a parametrized curve
    """

    def __init__(self, ts, xs=None, xf=None, init_nat_param=True):
        self.ts = ts
        self.tmax = np.max(ts)
        self.tmin = np.min(ts)
        if xf is not None:
            self.xs = xf(ts)
        elif xs is not None:
            self.xs = xs
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
        return np.linspace(lstart, lend, int((lend-lstart)/seg_size))
    
    def get_section_by_t(self, tstart, tend, seg_size=None):
        """
        returns an array of equally spaced l values corresponding to values from tstart to tend with approx segment size of seg_size
        """
        return self.get_section_by_l(self.l_of_t(tstart), self.l_of_t(tend), seg_size=seg_size)

class CircularCamRotater:
    def __init__(self, frame, omega, rv0 = np.array([1.0,1.0,1.1]), rotvec = np.array([1.0,1.0,0.7]), upVec = np.array([0,0,1])):
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


class VectorField(VMobject):
    def __init__(
        self,
        # Vectorized function: Takes in an array of coordinates, returns an array of outputs.
        func,
        # Typically a set of Axes or NumberPlane
        coordinate_system,
        density: float = 2.0,
        magnitude_range = None,
        color = None,
        color_map_name = "3b1b_colormap",
        color_map = None,
        stroke_opacity: float = 1.0,
        stroke_width: float = 3,
        tip_width_ratio: float = 4,
        tip_len_to_width: float = 0.01,
        max_vect_len: float | None = None,
        max_vect_len_to_step_size: float = 0.8,
        flat_stroke: bool = False,
        norm_to_opacity_func=None,  # TODO, check on this
        **kwargs
    ):
        self.func = func
        self.coordinate_system = coordinate_system
        self.stroke_width = stroke_width
        self.tip_width_ratio = tip_width_ratio
        self.tip_len_to_width = tip_len_to_width
        self.norm_to_opacity_func = norm_to_opacity_func

        # Search for sample_points
        self.sample_coords = get_sample_coords(coordinate_system, density)
        self.update_sample_points()

        if max_vect_len is None:
            step_size = get_norm(self.sample_points[1] - self.sample_points[0])
            self.max_displayed_vect_len = max_vect_len_to_step_size * step_size
        else:
            self.max_displayed_vect_len = max_vect_len * coordinate_system.get_x_unit_size()

        # Prepare the color map
        if magnitude_range is None:
            max_value = max(map(get_norm, func(self.sample_coords)))
            magnitude_range = (0, max_value)

        self.magnitude_range = magnitude_range

        if color is not None:
            self.color_map = None
        else:
            self.color_map = color_map or get_color_map(color_map_name)

        self.init_base_stroke_width_array(len(self.sample_coords))

        super().__init__(
            stroke_opacity=stroke_opacity,
            flat_stroke=flat_stroke,
            **kwargs
        )
        self.set_stroke(color, stroke_width)
        self.update_vectors()

    def init_points(self):
        n_samples = len(self.sample_coords)
        self.set_points(np.zeros((8 * n_samples - 1, 3)))
        self.set_joint_type('no_joint')

    def get_sample_points(
        self,
        center: np.ndarray,
        width: float,
        height: float,
        depth: float,
        x_density: float,
        y_density: float,
        z_density: float
    ) -> np.ndarray:
        to_corner = np.array([width / 2, height / 2, depth / 2])
        spacings = 1.0 / np.array([x_density, y_density, z_density])
        to_corner = spacings * (to_corner / spacings).astype(int)
        lower_corner = center - to_corner
        upper_corner = center + to_corner + spacings
        return cartesian_product(*(
            np.arange(low, high, space)
            for low, high, space in zip(lower_corner, upper_corner, spacings)
        ))

    def init_base_stroke_width_array(self, n_sample_points):
        arr = np.ones(8 * n_sample_points - 1)
        arr[4::8] = self.tip_width_ratio
        arr[5::8] = self.tip_width_ratio * 0.5
        arr[6::8] = 0
        arr[7::8] = 0
        self.base_stroke_width_array = arr

    def set_sample_coords(self, sample_coords):
        self.sample_coords = sample_coords
        return self

    def update_sample_points(self):
        self.sample_points = self.coordinate_system.c2p(*self.sample_coords.T)

    def update_vectors(self):
        tip_width = self.tip_width_ratio * self.stroke_width
        tip_len = self.tip_len_to_width * tip_width

        # Outputs in the coordinate system
        outputs = (self.func(self.sample_coords)-self.magnitude_range[0])/(self.magnitude_range[1]-self.magnitude_range[0])
        output_norms = np.linalg.norm(outputs, axis=1)[:, np.newaxis]

        # Corresponding vector values in global coordinates
        out_vects = self.coordinate_system.c2p(*outputs.T) - self.coordinate_system.get_origin()
        out_vect_norms = np.linalg.norm(out_vects, axis=1)[:, np.newaxis]
        unit_outputs = np.zeros_like(out_vects)
        np.true_divide(out_vects, out_vect_norms, out=unit_outputs, where=(out_vect_norms > 0))

        # How long should the arrows be drawn, in global coordinates
        max_len = self.max_displayed_vect_len
        if max_len < np.inf:
            drawn_norms = max_len * np.tanh(out_vect_norms / max_len)
        else:
            drawn_norms = out_vect_norms

        # What's the distance from the base of an arrow to
        # the base of its head?
        dist_to_head_base = np.clip(drawn_norms - tip_len, 0, np.inf)  # Mixing units!

        # Set all points
        points = self.get_points()
        points[0::8] = self.sample_points - drawn_norms * unit_outputs / 2
        points[2::8] = self.sample_points + dist_to_head_base * unit_outputs - drawn_norms * unit_outputs / 2
        points[4::8] = points[2::8]
        points[6::8] = self.sample_points + drawn_norms * unit_outputs / 2
        for i in (1, 3, 5):
            points[i::8] = 0.5 * (points[i - 1::8] + points[i + 1::8])
        points[7::8] = points[6:-1:8]

        # Adjust stroke widths
        width_arr = self.stroke_width * self.base_stroke_width_array
        width_scalars = np.clip(drawn_norms / tip_len, 0, 1)
        width_scalars = np.repeat(width_scalars, 8)[:-1]
        self.get_stroke_widths()[:] = width_scalars * width_arr

        # Potentially adjust opacity and color
        if self.color_map is not None:
            self.get_stroke_colors()  # Ensures the array is updated to appropriate length
            low, high = 0.0,1.0#self.magnitude_range
            self.data['stroke_rgba'][:, :3] = self.color_map(
                inverse_interpolate(low, high, np.repeat(output_norms, 8)[:-1])
            )[:, :3]

        if self.norm_to_opacity_func is not None:
            self.get_stroke_opacities()[:] = self.norm_to_opacity_func(
                np.repeat(output_norms, 8)[:-1]
            )

        self.note_changed_data()
        return self