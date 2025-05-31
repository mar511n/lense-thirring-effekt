# run using: manimgl manim_scenes.py OrbitalPlane --config_file config.yml --show_animation_progress

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from manimlib import *
from typing import Any
import numpy as np
import manim_tools as mt
import lense_thirring_tools as ltt
#import scipy.integrate as spint
#import scipy.interpolate as spinter
#import cv2

DARK_MODE: bool = True
LIGHT_MODE: bool = False # experimental
OFFBLACK = rgb_to_color(hex_to_rgb("#121317"))
OFFWHITE = rgb_to_color(hex_to_rgb("#F0F8FF"))
Theme = DARK_MODE
BACKCOL = OFFBLACK if Theme else OFFWHITE
FRONTCOL = OFFWHITE if Theme else OFFBLACK
def set_theme(theme):
    global Theme, BACKCOL, FRONTCOL, OFFWHITE, OFFBLACK
    Theme = theme
    BACKCOL = OFFBLACK if Theme else OFFWHITE
    FRONTCOL = OFFWHITE if Theme else OFFBLACK

TITLE_FONT_SIZE = 52
CONTENT_FONT_SIZE = 0.75 * TITLE_FONT_SIZE

def make_orbit(face_camera,omega = np.pi/8):
    axes = ThreeDAxes(
        x_range=(-3,3,1),
        y_range=(-3,3,1),
        z_range=(-3,3,1),
    )
    axes.apply_depth_test()
    
    ts = np.linspace(0,np.pi*2,100)
    xs = np.array([[1.5*np.cos(phi), 1.5*np.sin(phi),0] for phi in ts])
    pc = ltt.ParametricCurve(ts,xs)
    cd = mt.CurveDrawer([pc],fixed_color=FRONTCOL)
    cd.update_graphics(arrow_tss=np.array([0,0.25,0.5,0.75]),arrow_length=0.2)
    orbit_plane = Group(Disk3D(radius=1.5,color=GREY_D),cd)#,Circle(radius=1.5,stroke_color=FRONTCOL))
    orbit_plane.stretch(1.3,0)
    orbit_size = orbit_plane.get_shape()
    L_arr = mt.Arrow3D(end=np.array([0,0,2]),color=BLUE_D,shaft_width=0.03,tip_length=0.2)
    j_arr = mt.Arrow3D(end=np.array([orbit_size[0]/2,0,0]),color=BLUE_D,shaft_width=0.03,tip_length=0.2)
    i_arr = mt.Arrow3D(end=np.array([0,-orbit_size[1]/2,0]),color=BLUE_D,shaft_width=0.03,tip_length=0.2)
    k_arr = mt.Arrow3D(start=np.array([0,-3,0]),end=np.array([0,3.2,0]),color=RED_D,shaft_width=0.03,tip_length=0.2)

    orbit = Group(orbit_plane,L_arr,j_arr,i_arr)
    orbit.rotate(np.pi/16,(-1,0,0),(0,0,0))
    orbit.rotate(np.pi/12,(0,1,0),(0,0,0))
    orbit.rotate(np.pi/8,(0,0,1),(0,0,0))
    orbit.apply_depth_test()

    L_vec = TexText(r'$\vec{L}$',fill_color=BLUE_C)
    L_vec.move_to(L_arr.get_center()*2.1+OUT*0.3)
    
    j_vec = TexText(r'$\vec{j}$',fill_color=BLUE_C)
    j_vec.move_to(j_arr.get_center()*2.1+OUT*0.3)
    
    i_vec = TexText(r'$\vec{i}$',fill_color=BLUE_C)
    i_vec.move_to(i_arr.get_center()*2.1+OUT*0.3)
    
    Lv = L_arr.get_center()*2
    jv = j_arr.get_center()*2
    iv = i_arr.get_center()*2
    a_p = TexText(r'$a_{\mathrm{p}}$',fill_color=BLUE_C)
    a_p.move_to(jv*0.5+DOWN*0.4)

    k_vec = TexText(r'$\vec{K}$',fill_color=RED_C)
    k_vec.move_to((0,3.2,0)+OUT*0.4)

    k_proj = np.dot(jv,[0,1,0])*np.array([0,1,0])
    x_obs = mt.Arrow3D((0,0,0),k_proj,color=GREEN_C,shaft_width=0.04,tip_length=0.2)
    proj_l = Line3D(jv,k_proj,width=0.06,color=GREEN_A,shading=(0,0,0),resolution=(10,10))

    x_obs_t = TexText(r'$x_{\mathrm{obs}}$',fill_color=GREEN_C)
    x_obs_t.move_to(k_proj*1.05 + RIGHT*0.5)
    

    precession_circ = Circle(stroke_color=BLUE_D,radius=np.sqrt(Lv[0]**2 + Lv[1]**2)).move_to((0,0,Lv[2]))

    for vmob in [L_vec,j_vec,i_vec,a_p,k_vec,x_obs_t]:
        vmob.save_state()

    def updater(obj,dt):
        nonlocal L_vec, Lv, j_vec, jv, i_vec, iv, a_p, x_obs, k_proj, proj_l, x_obs_t
        obj.rotate(omega*dt,(0,0,1),(0,0,0))
        Lv = rotate_vector(Lv, omega*dt,(0,0,1))
        jv = rotate_vector(jv, omega*dt,(0,0,1))
        iv = rotate_vector(iv, omega*dt,(0,0,1))
        L_vec.saved_state.move_to(Lv*1.05+OUT*0.3)
        j_vec.saved_state.move_to(jv*1.1+OUT*0.3)
        i_vec.saved_state.move_to(iv*1.1+OUT*0.3)
        a_p.saved_state.move_to(jv*0.5+DOWN*0.4)
        k_proj = np.dot(jv,[0,1,0])*np.array([0,1,0])
        x_obs.become(mt.Arrow3D((0,0,0),k_proj,color=GREEN_D,shaft_width=0.04,tip_length=0.2))
        proj_l.become(Line3D(jv,k_proj,width=0.06,color=GREEN_A,shading=(0,0,0),resolution=(10,10)))
        x_obs_t.saved_state.move_to(k_proj+k_proj/k_proj[1]*0.25)
    updater(orbit,0)
    for vmob in [L_vec,j_vec,i_vec,a_p,k_vec,x_obs_t]:
        face_camera(vmob,0)
    return ((orbit,k_arr,x_obs,proj_l),(axes,precession_circ),(L_vec,j_vec,i_vec,a_p,k_vec,x_obs_t)), updater

class OrbitalPlane(ThreeDScene):
    def __init__(
            self,
            window = None,
            camera_config: dict = dict(),
            file_writer_config: dict = dict(),
            skip_animations: bool = False,
            always_update_mobjects: bool = False,
            start_at_animation_number: int | None = None,
            end_at_animation_number: int | None = None,
            show_animation_progress: bool = False,
            leave_progress_bars: bool = False,
            preview_while_skipping: bool = True,
            presenter_mode: bool = False,
            default_wait_time: float = 1.0,
    ):
        camera_config['background_color'] = BACKCOL
        camera_config['light_source_position'] = np.array([10, -10, 10])
        super().__init__(window, camera_config, file_writer_config, skip_animations, always_update_mobjects, start_at_animation_number, end_at_animation_number, show_animation_progress, leave_progress_bars, preview_while_skipping, presenter_mode, default_wait_time)
    
    def construct(self):
        def face_camera(mob:Mobject, dt):
            mob.become(mob.saved_state)
            mob.rotate(self.frame.get_phi(), axis=RIGHT)
            mob.rotate(self.frame.get_theta(), axis=OUT)
        def let_obj_face_cam(mob:Mobject):
            mob.save_state()
            mob.add_updater(face_camera)
        self.frame.reorient(45,48,0)
        (mobjs,vmobjs,fvmobjs), updater = make_orbit(face_camera)
        self.play(*[FadeIn(mob) for mob in mobjs],*[Write(vmob) for vmob in vmobjs], *[Write(vmob) for vmob in fvmobjs])
        self.wait(4.0)
        mobjs[0].add_updater(updater)
        for fvmob in fvmobjs: fvmob.add_updater(face_camera)
        #self.add(orbit,k_arr,x_obs,proj_l,axes,L_vec,j_vec,i_vec,a_p,k_vec,x_obs_t,precession_circ)
        return super().construct()