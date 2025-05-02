from manimlib import *
from manim_slides import Slide
from typing import Any
import numpy as np
import scipy.integrate as spint
import scipy.interpolate as spinter
import manim_tools as mt
import lense_thirring_tools as ltt
import cv2
import re

"""
Strukturierung:

    Einleitende Dinge zu ART (Geodäten & Metrik, Einsteingl. -> Metrik)
        Animation von gekrümmter Fläche, Metrik an einem Punkt, Geodäte
        Wie bestimmt man die Metrik?? -> EFG (Problem: nichtlinear, gekoppelt)
    Linearisierung Einsteingl. (Annahme h<<1 & tau=t & v<<c sagen und g=mu + h)
        resultierende Gl. <-> e-dynamik (erst maxwell, dann Coulomb-Kraft (ersetzen der B,E Felder))
    Problem einer rotierenden Kugelmasse (dichte und stromdichte hinschreiben) (sagen, dass Lösung wie schon in E-dynamik is)
        Darstellung der E,B Felder
        ((Darstellung der Trajektorien (ist schwierig/nicht machbar, da die Auslenkungen nicht sichtbar sind)))
?       Alternative Darstellung über die Raumzeit (auch schwierig)
?   Gravity Probe B
    Paper 1
    Paper 2

1 teaser
2 inhaltsverzeichnis
3 inhalt
"""
PresentationTitle = 'Lense-Thirring-Effekt'
PresentationInfo = r"""
{
Vortrag im Hauptseminar SoSe 2025 \\
Marvin Henke - \today \\
Betreuer: Dr. Nikodem Szpak \\
}
"""
PresentationContactInfo = 'marvin.henke@stud.uni-due.de'
PresentationContents = ('Metrik und Geodäten', 'Einsteinsche Feldgleichungen', 'Gravitoelektromagnetismus', 'Rotierende Kugelmasse', 'EM-Felder', 'Gravity Probe B', 'Paper')

DARK_MODE: bool = True
LIGHT_MODE: bool = False # highly experimental
OFFBLACK = rgb_to_color(hex_to_rgb("#121317"))
OFFWHITE = rgb_to_color(hex_to_rgb("#F0F8FF"))
Theme = DARK_MODE
BACKCOL = OFFBLACK if Theme else OFFWHITE
FRONTCOL = OFFWHITE if Theme else OFFBLACK

TITLE_FONT_SIZE = 52
CONTENT_FONT_SIZE = 0.75 * TITLE_FONT_SIZE

default_kwargs_vmobj = {
    'fill_color' : FRONTCOL,
    'stroke_color' : FRONTCOL,
}
default_kwargs_text = {
    'base_color' : FRONTCOL,
    'z_index' : 100,
}
class VMobject(VMobject):
    def __init__(self, color = None, fill_color = BACKCOL, fill_opacity = 0, stroke_color = FRONTCOL, stroke_opacity = 1, stroke_width = DEFAULT_STROKE_WIDTH, stroke_behind = False, background_image_file = None, long_lines = False, joint_type = "auto", flat_stroke = False, scale_stroke_with_zoom = False, use_simple_quadratic_approx = False, anti_alias_width = 1.5, fill_border_width = 0, **kwargs):
        super().__init__(color, fill_color, fill_opacity, stroke_color, stroke_opacity, stroke_width, stroke_behind, background_image_file, long_lines, joint_type, flat_stroke, scale_stroke_with_zoom, use_simple_quadratic_approx, anti_alias_width, fill_border_width, **kwargs)

class TexText(TexText):
    def __init__(self, *tex_strings, font_size = CONTENT_FONT_SIZE, alignment = r"\centering", template = "custom", additional_preamble = "", tex_to_color_map = dict(), t2c = dict(), isolate = [], use_labelled_svg = True, **kwargs):
        kwargs = default_kwargs_text | default_kwargs_vmobj | kwargs
        super().__init__(*tex_strings, font_size=font_size, alignment=alignment, template=template, additional_preamble=additional_preamble, tex_to_color_map=tex_to_color_map, t2c=t2c, isolate=isolate, use_labelled_svg=use_labelled_svg, **kwargs)

class BulletedList(BulletedList):
    def __init__(self, *items, buff = MED_LARGE_BUFF, aligned_edge = LEFT, **kwargs):
        kwargs = default_kwargs_text | default_kwargs_vmobj | kwargs
        super().__init__(*items, buff=buff, aligned_edge=aligned_edge, **kwargs)

class Integer(Integer):
    def __init__(self, number = 0, num_decimal_places = 0, **kwargs):
        kwargs = default_kwargs_vmobj | kwargs
        super().__init__(number, num_decimal_places, **kwargs)

class LenseThirringGL(Slide):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.high_quality = False
        kwargs['show_animation_progress'] = True
        #kwargs['leave_progress_bars'] = True
        kwargs['camera_config'] = {'background_color':BACKCOL}
        kwargs['camera_config']['light_source_position'] = np.array([10, -10, 10])
        #kwargs['start_at_animation_number'] = 18
        #kwargs['end_at_animation_number'] = 22
        print(kwargs)
        super().__init__(*args, **kwargs)
        if self.high_quality:
            self.samples = 16
        self.max_duration_before_split_reverse = 100
    def construct(self):
        # basic setup
        #   canvas objects (are not wiped)
        self.canvas_objs = [self.frame]
        #   skip video rendering for faster build
        self.rendervideos = self.high_quality
        #   manual reference of slide number
        self.slide_number_val = 0
        #   lense-thirring parameters (earth mass, maximal possible rotation speed of the earth)
        ltt.set_params_lense_thirring(mass=6.96e-10,omega=2.64e-5,radius=1)
        sphere_omega = 2*np.pi/2
        #   fix_in_frame objs z_index
        self.z_idx_fix = default_kwargs_text['z_index']
        #   set 2d screen space offset of 3d objects
        self.offset_3d = np.array([2.2,0.2])
        self.camera.uniforms['shift_screen_space'] = self.offset_3d
        #   predefined colors for certain tex parts
        symCols = {
            r'\vec{r}':BLUE_D,
            r'{p}':RED,
            r'{g}':YELLOW_D,
            r'\bm{g}':YELLOW_D,
            r'{u}':DARK_BROWN,
            r'{v}':LIGHT_BROWN,
            r'\Gamma':PURPLE_D
        }
        #   function to align Mobjs on the left below the title
        def align_mobjs(mobjs):
            for i,mobjl in enumerate(mobjs):
                last = mobjs[i-1][0] if i>0 else self.slide_title
                for mobj in mobjl:
                    mobj.fix_in_frame()
                    mobj.next_to(last.get_corner(DL),DOWN,aligned_edge=LEFT)
        #   call let_obj_face_cam on a mobj to make it face the camera
        def face_camera(mob:Mobject, dt):
            newtheta = self.frame.get_theta()
            newphi = self.frame.get_phi()
            mob.rotate(-mob.theta, axis=OUT)
            mob.rotate(-mob.phi, axis=RIGHT)
            mob.rotate(newphi, axis=RIGHT)
            mob.rotate(newtheta, axis=OUT)
            mob.theta = newtheta
            mob.phi = newphi
        def let_obj_face_cam(mob:Mobject):
            mob.theta = 0.0
            mob.phi = 0.0
            mob.add_updater(face_camera)
        #   cam rotater
        camRot = mt.CircularCamRotater(self.frame, 2*np.pi/4,rv0 = np.array([0.0,-1.0,1.6]), rotvec = np.array([0.0,-1.0,1.3]))
        self.pause()


        # Titlepage (0)
        render_height = 0.6
        TextBox = Rectangle(height=FRAME_HEIGHT*(1-render_height), width=FRAME_WIDTH)
        TextBox.set_style(fill_color=BACKCOL, fill_opacity=1, stroke_width=0)
        TextBox.to_edge(DOWN,buff=0)
        TextBox.fix_in_frame()
        self.add(TextBox)

        self.slide_title = TexText(rf"\textbf{{{PresentationTitle}}}", font_size=TITLE_FONT_SIZE)
        self.slide_title.next_to(TextBox.get_corner(UL), RIGHT+DOWN, MED_LARGE_BUFF)
        self.slide_title.fix_in_frame()
        self.canvas_objs.append(self.slide_title)

        presentation_info = TexText(PresentationInfo, alignment=None)
        presentation_info.next_to(TextBox.get_corner(DL), RIGHT+UP, MED_LARGE_BUFF)
        presentation_info.fix_in_frame()

        background_render = ImageMobject('./assets/lense_thirring.png',height=FRAME_HEIGHT)
        background_render.shift(UP*FRAME_HEIGHT*(1-render_height)/2)
        background_render.fix_in_frame()
        self.add(background_render)
        self.bring_to_back(background_render)

        ude_logo = SVGMobject('./assets/logo/logo_claim_rgb.svg' if Theme else './assets/logo/logo_claim_rgb_neg.svg', height=1, z_index=self.z_idx_fix)
        ude_logo.to_corner(UR, buff=0)
        ude_logo.shift(DOWN*0.1)
        ude_logo.fix_in_frame()
        self.canvas_objs.append(ude_logo)

        self.play(Write(self.slide_title), Write(ude_logo), Write(presentation_info))#, Write(contact_info))
        self.pause()


        # Layout stuff (1)
        #   Kontaktdaten
        contact_info = TexText(PresentationContactInfo, font_size=CONTENT_FONT_SIZE*0.6).to_corner(DL,buff=MED_SMALL_BUFF)#.shift(RIGHT*(DEFAULT_MOBJECT_TO_EDGE_BUFF-SMALL_BUFF))
        contact_info.fix_in_frame()
        self.canvas_objs.append(contact_info)
        #   Präsentationstitel
        presentation_title = TexText(PresentationTitle, font_size=CONTENT_FONT_SIZE*0.6).align_to(contact_info, UP)
        presentation_title.fix_in_frame()
        self.canvas_objs.append(presentation_title)
        #   Seitennummer
        self.slide_number_val = 1
        self.slide_number = Integer(1, font_size=CONTENT_FONT_SIZE*0.6).to_corner(DR,buff=MED_SMALL_BUFF)
        self.slide_number.fix_in_frame()
        self.canvas_objs.append(self.slide_number)
        self.play(FadeOut(presentation_info),FadeOut(TextBox),FadeOut(background_render), Write(contact_info,run_time=0.5), Write(presentation_title,run_time=0.5), Write(self.slide_number,run_time=0.5), self.slide_title.animate.to_corner(UL))
        self.pause()


        # Inhaltsverzeichnis (2-8)
        Inhalt = BulletedList(*PresentationContents, buff=MED_SMALL_BUFF)
        Inhalt.to_edge(LEFT)
        Inhalt.fix_in_frame()
        for istr in range(len(PresentationContents)):
            self.play(Write(Inhalt[istr],run_time=0.5))
            if istr < len(PresentationContents)-1:
                self.pause()
        self.pause()

        # Metrik & Geodätengleichung (9-17)
        self.setup_new_slide(title='Metrik und Geodäten', cleanup=True)
        self.pause()


        #   Euklidisch (10-13)
        fläche_text = TexText(r'Fläche')
        rfunc = TexText(r'$\vec{r} : \mathbb{R}^2 \rightarrow \mathbb{R}^3$',isolate=[r'\vec{r}']).set_color_by_tex_to_color_map(symCols)
        rfunc_euclid = TexText(r'$\vec{r}({u},{v}) = ({u},{v},0)$',isolate=[r'{u}',r'{v}',r'\vec{r}']).set_color_by_tex_to_color_map(symCols)
        metrik_text = TexText(r'Metrik').fix_in_frame()
        g_tensor = TexText(r'${g}_{\mu \nu} = \partial_{\mu} \vec{r}\cdot\partial_{\nu} \vec{r}$', isolate=[r'{g}', r'\vec{r}']).set_color_by_tex_to_color_map(symCols)
        g_euclid = VGroup(TexText(r'$\bm{g} = $',isolate=r'\bm{g}').set_color_by_tex_to_color_map(symCols,only_isolated=True),TexText(r'$\left[\begin{array}{c}1 \quad 0 \\0 \quad 1 \\\end{array}\right]$')).arrange()
        geod_gl_text = TexText(r'Geodätengleichung')
        geod_gl_t1 = TexText(r'$\frac{\mathrm{d}^2 {p}^{\lambda}}{\mathrm{d} \tau^2} = $',isolate=[r'{p}']).set_color_by_tex_to_color_map(symCols)
        geod_gl_t2 = TexText(r'$-\Gamma^{\lambda}_{\mu \nu}[\bm{g}(\vec{r})] \frac{\mathrm{d} {p}^{\mu}}{\mathrm{d} \tau} \frac{\mathrm{d} {p}^{\nu}}{\mathrm{d} \tau}$',isolate=[r'{p}',r'\bm{g}',r'\vec{r}',r'\Gamma']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        geod_gl_eucild = VGroup(geod_gl_t1.copy(),TexText(r'$0$')).arrange()
        geod_gl = VGroup(geod_gl_t1,geod_gl_t2).arrange()
        itms = [(fläche_text,),(rfunc,rfunc_euclid),(metrik_text,),(g_tensor,),(g_euclid,),(geod_gl_text,),(geod_gl_eucild,geod_gl)]
        align_mobjs(itms)

        ltt.set_params_gaußian_surface(1/np.sqrt(2),A=0.0)
        grid_nc,_ = mt.get_grid_surface(uv_func=lambda u,v: [u,v,v*0], u_range=(-3,3), v_range=(-3,3), grid_size=(8,8), grid_col=FRONTCOL)
        ts,zs = ltt.get_geodesic(6, np.array([3,0.5,-1,0]),tol=1e-9,accF=ltt.acc_gaußian_surface,check_break=lambda t,r: np.abs(r[0])>3 or np.abs(r[1])>3)
        pcd_nc = mt.CurveDrawer([mt.ParametricCurve(ts,[[uv[0],uv[1],0] for uv in zs[:,:2]])],fixed_color=ORANGE)
        pcd_nc.update_graphics()
        dot = Sphere(radius=0.05,color=RED,shading=(0,0,0)).apply_depth_test()
        u_label = TexText(r'${u}$').next_to(grid_nc.get_bottom(),DOWN).set_color_by_tex_to_color_map(symCols)
        v_label = TexText(r'${v}$').next_to(grid_nc.get_left(),LEFT).set_color_by_tex_to_color_map(symCols)
        self.play(Write(grid_nc), Write(pcd_nc), Write(u_label), Write(v_label))
        self.pause()


        self.play(Write(fläche_text),Write(rfunc_euclid))
        self.pause()


        self.play(Write(metrik_text),Write(g_tensor), Write(g_euclid))
        self.pause(auto_next=True)


        self.play(Write(geod_gl_text),Write(geod_gl_eucild))
        self.pause(loop=True)


        #   Animation euklidischer Raum (14)
        self.add(dot)
        camRot.startUpdating()
        self.play(MoveAlongPath(dot,pcd_nc), run_time=4.0,rate_func=linear)
        camRot.stopUpdating()
        self.remove(dot)
        self.pause(auto_next=True)


        #   Gekrümmter Raum (15-17)
        ltt.set_params_gaußian_surface(1/np.sqrt(2),A=2.0)
        _, surface = mt.get_grid_surface(uv_func=lambda u,v: [u,v,ltt.z_gaußian_surface(u,v)], u_range=(-3,3), v_range=(-3,3), grid_size=(8,8), grid_col=FRONTCOL)
        surface.set_color_by_rgba_func(lambda r: get_color_map('viridis')(r[2]/ltt.z_gaußian_surface(0,0)))
        ts,zs = ltt.get_geodesic(10, np.array([3,0.5,-1,0]),tol=1e-9,accF=ltt.acc_gaußian_surface,check_break=lambda t,r: np.abs(r[0])>3 or np.abs(r[1])>3)
        pcd = mt.CurveDrawer([mt.ParametricCurve(ts,[[uv[0],uv[1],ltt.z_gaußian_surface(uv[0],uv[1])+1e-2] for uv in zs[:,:2]])],fixed_color=ORANGE)
        pcd.update_graphics()
        metric = DecimalMatrix(((1,0),(0,1)),num_decimal_places=2).scale(0.7).fix_in_frame()
        metric.next_to(g_euclid[1].get_corner(UL),RIGHT,aligned_edge=UP,buff=0)
        metric.set_color(FRONTCOL)
        def m_update(mob,dt):
            nonlocal dot
            pos = dot.get_center()
            g = ltt.metric_gaußian_surface(pos[0],pos[1]).flatten()
            for i in range(4):
                mob.elements[i].set_value(g[i])
        self.play(FadeIn(surface), grid_nc.animate.apply_function(lambda r: [r[0],r[1],ltt.z_gaußian_surface(r[0],r[1])+1e-2]), ReplacementTransform(pcd_nc, pcd), ReplacementTransform(rfunc_euclid,rfunc), ReplacementTransform(g_euclid[1],metric), ReplacementTransform(geod_gl_eucild[1],geod_gl[1]))
        self.wait(0.04)
        self.pause(loop=True)


        metric.add_updater(m_update)
        self.add(dot)
        camRot.startUpdating()
        self.play(MoveAlongPath(dot,pcd), run_time=4.0,rate_func=linear)
        camRot.stopUpdating()
        self.pause()


        # EFGl mit Analogie 2D Fläche eingebettet in 3D Raum -> 4D Fläche (18-21)
        self.setup_new_slide(title='Einsteinsche Feldgleichungen',cleanup=True)
        text1 = TexText(r'2D Fläche $\rightarrow$ 4D Mannigfaltigkeit')
        text2 = TexText(r'Koordinaten $(ct,x,y,z)$ $\Rightarrow$ $\bm{g}\in\mathbb{R}^{4\times 4}$',isolate=[r'\bm{g}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        efe = TexText(r'Feldgleichungen: ',r'$R_{\mu \nu} - \frac{1}{2} {g}_{\mu \nu} R = \frac{8 \pi G}{c^4} T_{\mu \nu}$',isolate=[r'\bm{g}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        itms = [(text1,),(text2,),(efe,)]
        align_mobjs(itms)
        self.play(Write(text1))
        self.pause()


        self.play(Write(text2))
        self.pause()


        self.play(Write(efe))
        self.pause()

        # EM-Felder (22-27)
        #   Formeln (22-23)
        self.setup_new_slide(title='EM-Felder',cleanup=True)
        bfield_formula = TexText(r'$\vec{B}=\frac{1}{r^3}\left[\vec{S} - \frac{3(\vec{S}\cdot\vec{r})}{r^2}\vec{r}\right]$')
        bfield_formula.next_to(self.slide_title.get_corner(DL), DOWN, aligned_edge=LEFT, buff=1.2*DEFAULT_MOBJECT_TO_MOBJECT_BUFF)
        bfield_formula.fix_in_frame()
        efield_formula = TexText(r'$\vec{E}=-\frac{M \vec{r}}{r^3}$')
        efield_formula.next_to(bfield_formula.get_corner(DL), DOWN, aligned_edge=LEFT, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFF)
        efield_formula.fix_in_frame()
        self.play(Write(bfield_formula,run_time=0.5),Write(efield_formula,run_time=0.5))
        self.pause(auto_next=True)
        

        formula_box = SurroundingRectangle(bfield_formula, color=ORANGE)
        #   3D Axen & CamRot & Kugel
        axes = ThreeDAxes(
            x_range=(-3,3,1),
            y_range=(-3,3,1),
            z_range=(-3,3,1),
        )
        self.canvas_objs.append(axes)
        axes.apply_depth_test(recurse=True)
        camRot = mt.CircularCamRotater(self.frame, 2*np.pi/4,rv0 = np.array([1.0,-1.0,1.1]), rotvec = np.array([1.0,-1.0,0.7]))

        sphere = Sphere(radius=ltt.R)
        day_texture = "./assets/Whole_world_-_land_and_oceans.jpg"
        night_texture = "./assets/The_earth_at_night.jpg"
        sphere = TexturedSurface(sphere, day_texture, night_texture)
        def rotater(obj,dt):
            nonlocal sphere_omega
            obj.rotate(dt*sphere_omega)
        sphere.add_updater(rotater)
        self.canvas_objs.append(sphere)

        bfmax = np.linalg.norm(ltt.bfield(np.array([[0,0,ltt.R]])))
        efmax = np.linalg.norm(ltt.efield(np.array([[0,0,ltt.R]])))

        #   B-Feld (24)
        rs = [1, 2, 1]
        zs = [-2, 0, 2]
        phis = np.arange(0,2*np.pi,np.pi/4)
        startPoints = np.array([[rs[zi]*np.cos(phi),rs[zi]*np.sin(phi),zs[zi]] for phi in phis for zi in range(len(zs))])
        bounds = np.array([[-2.5,-2.5,-2.5],[2.5,2.5,2.5]])
        sls_b = mt.StreamLines(fieldf=lambda t,x: ltt.bfield(np.array([x])), startPoints=startPoints, boundary=bounds, system_timescale=1/bfmax, vmax=bfmax)
        self.play(Write(formula_box),Write(axes),FadeIn(sphere),Write(sls_b))
        self.pause(loop=True)

        
        #   B-Feld Animation (25)
        sls_b.startUpdating(timeScaleF=0.25)
        camRot.startUpdating()
        self.wait(4.0)
        self.pause(auto_next=True)


        #   E-Feld (26)
        sls_b.stopUpdating()
        camRot.stopUpdating()
        sls_e = mt.StreamLines(fieldf=lambda t,x: ltt.efield(np.array([x])), boundary=bounds, system_timescale=1/efmax, vmax=efmax)
        self.play(formula_box.animate.surround(efield_formula), FadeOut(sls_b), Write(sls_e))
        self.pause(loop=True)


        #   E-Feld Animation (27)
        sls_e.startUpdating(timeScaleF=0.25)
        camRot.startUpdating()
        self.wait(4.0)
        self.pause()


    def get_non_canvas_mobjs(self):
        return [mobj for mobj in self.get_top_level_mobjects() if mobj not in self.canvas_objs]

    def wait(self,dt=1.0):
        mobj = Mobject()
        self.play(Animation(mobj,run_time=dt),rate_func=linear)
        self.remove(mobj)

    def pause(self,notes="",loop=False, auto_next=False):
        #self.wait(0.1)
        self.next_slide(loop=loop,notes=notes,auto_next=auto_next)

    def next_slide_number_animation(self):
        self.slide_number_val += 1
        return self.slide_number.animate(run_time=0.5).set_value(
            self.slide_number_val
        )

    def next_slide_title_animation(self, title):
        newT = TexText(rf"\textbf{{{title}}}", font_size=TITLE_FONT_SIZE).move_to(self.slide_title, UP).align_to(self.slide_title, LEFT)
        newT.fix_in_frame()
        return Transform(
            self.slide_title,
            newT,
            run_time=0.5,
        )

    def setup_new_slide(self, title, cleanup=False, contents=None):
        if cleanup:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
                self.wipe(
                    self.get_non_canvas_mobjs(),
                    contents if contents else [],
                    return_animation=True,
                ),
            )
        else:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
            )
    
    def make_video(self, videopath, update_period=1, no_video=False):
        cap = cv2.VideoCapture(videopath)
        vid_fi = 1
        vid_us = 0
        vid_has, frame = cap.read()
        cv2.imwrite(f"/tmp/hauptseminar_gl_{vid_fi}.png",frame)
        frame_img = ImageMobject(f"/tmp/hauptseminar_gl_{vid_fi}.png")
        frame_img.set_height(self.frame.get_height())
        def frame_updater(obj, dt):
            nonlocal vid_fi, vid_has, update_period, vid_us
            if vid_has and dt > 0:
                vid_us += 1
                if vid_us >= update_period:
                    vid_us = 0
                    frame_img = ImageMobject(f"/tmp/hauptseminar_gl_{vid_fi}.png")
                    frame_img.match_height(obj)
                    frame_img.match_width(obj)
                    frame_img.match_updaters(obj)
                    self.replace(obj, frame_img)
                    vid_has, frame = cap.read()
                    if vid_has:
                        vid_fi += 1
                        cv2.imwrite(f"/tmp/hauptseminar_gl_{vid_fi}.png",frame)
        if not no_video:
            frame_img.add_updater(frame_updater,call=False)
        return frame_img, cap