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
 ✓  Titelfolie
(✓) Inhaltsverzeichnis (und nebenan Darstellung der Trajektorien als Gaswolken)
 ✓  Einleitende Dinge zu ART (Geodäten & Metrik, Einsteingl. -> Metrik) (sagen, dass Geodäten lokal den Weg minimieren & aus Funktional der Länge hergeleitet werden können)
 ✓      Animation von gekrümmter Fläche, Metrik an einem Punkt, Geodäte
 ✓      Wie bestimmt man die Metrik?? -> EFG (Problem: nichtlinear, ableitungen von g)
 ✓  Linearisierung Einsteingl. (Annahme h<<1 & tau=t & v<<c sagen und g=mu + h)
 ✓      resultierende Gl. <-> Maxwell-Gl. (Kopplung Materie -> Metrik)
 ✓      Bewegungsgleichung aus Geodätengleichung (Kopplung Metrik -> Materie)
 ✓  Problem einer rotierenden Kugelmasse (dichte und stromdichte hinschreiben) (sagen, dass Lösung wie schon in E-dynamik is)
 ✓      Darstellung der E,B Felder (gilt auch für nicht homogene Objekte mit Drehimpuls)
 ✓      Einzelne ausgewählte Trajektorien zeigen
 ✓      Alternative Darstellung über die Raumzeit (-> xy-Ebene -> Präzession)
 ✓      Darstellung & Herleitung Präzession
    Gravity Probe B
    Akkretionsscheibe
    Pulsar

    
TODO:
self.pause überprüfen
rate functions (linear) überprüfen

Trajektorien: Erde kleiner machen (R->r), statt omega S benutzen; damit S gleich bleibt => (S_0 = 2/5*omega, S_1 = S_0 * (r/R)^2)
    alternativ: Nikodem überzeugen, dass R=1 wichtig ist, da sonst omega>1 => v > c, was nicht geht

Wichtig zu erwähnen:
    - Lense-Thirring-Effekt: Effekte der ART in erster Ordnung für eine rotierende Masse
    - Geodäten lassen sich aus Minimierung der Weglänge über Wirkungsintegral & Euler-Lagrange-gl. bestimmen
    - EFE sind 16 nichtlineare, gekoppelte Differentialgleichungen; durch Symmetrien -> 10 DGLs; (außerdem 4 Bianchi-Identitäten)
    - Linearisierung: Annahmen sagen; durch geschickte Wahl von Potentialen => Maxwell-Gleichungen (bis auf Konstanten)
    - Kugelmasse: Lösen ist analog zu E-Dynamik
    - E-Feld wie das einer negativen Punktladung
    - B-Feld ähnlich dem eines Kreisstroms
    - Ablenkung der Trajektorie durch Mitrotation
    - Raumzeit: für jeden Punkt auf dem Gitter wird die zeitliche Entwicklung dargestellt
    - Newton: Gravitationskraft <=> Raum fällt um uns herum
    - Präzession: wie ein Grashalm auf Wasser, bei dem die Strömungsgeschwindigkeiten an beiden Enden unterschiedlich sind
    - Genaue Betrachtung anhand von zwei Punkten auf dem Probekörper
"""

PresentationTitle = 'Lense-Thirring-Effekt'
PresentationInfo = r"""
{
Vortrag im Hauptseminar SoSe 2025 \\
Marvin Henke - 12. Juni 2025 \\
Betreuer: Dr. Nikodem Szpak \\
}
"""
PresentationContactInfo = 'marvin.henke@stud.uni-due.de'
PresentationContents = (
    (0, 'Allgemeine Relativitätstheorie'),
    (1, 'Metrik und Geodäten'),
    (1, 'Einsteinsche Feldgleichungen'),
    (1, 'Gravitoelektromagnetismus'),
    (0, 'Rotierende Kugelmasse'),
    (1, 'EM-Felder'),
    (1, 'Trajektorien'),
    (1, 'Präzession'),
    (0, 'Aktuelle Forschung'),
    (1, 'Gravity Probe B'),
    (1, 'Akkretionsscheibe eines supermassiven Schwarzen Lochs'),
    (1, 'Binärsystem aus Pulsar und weißem Zwerg'))

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
    def __init__(self, *tex_strings, font_size = CONTENT_FONT_SIZE, alignment = r"\centering", template = "custom", additional_preamble = "", tex_to_color_map = dict(), t2c = dict(), isolate = [], use_labelled_svg = True, fix_in_frame=True, **kwargs):
        kwargs = default_kwargs_text | default_kwargs_vmobj | kwargs
        super().__init__(*tex_strings, font_size=font_size, alignment=alignment, template=template, additional_preamble=additional_preamble, tex_to_color_map=tex_to_color_map, t2c=t2c, isolate=isolate, use_labelled_svg=use_labelled_svg, **kwargs)
        if fix_in_frame:
            self.fix_in_frame()

class BulletedList(VGroup):
    def __init__(
        self,
        *items: str,
        buff: float = SMALL_BUFF,
        aligned_edge = LEFT,
        **kwargs
    ):  
        labelled_content = [r'\item '+item[1] for item in items]
        tex_string = r'\begin{itemize}'+"\n"
        ci = items[0][0]
        for item in items:
            if item[0] > ci:
                tex_string += r'\begin{itemize}'+"\n"
            elif item[0] < ci:
                tex_string += r'\end{itemize}'+"\n"
            tex_string += r'\item '+item[1]+"\n"
            ci = item[0]
        for i in range(ci):
            tex_string += r'\end{itemize}'+"\n"
        tex_string += r'\end{itemize}'
        kwas1 = default_kwargs_text | default_kwargs_vmobj | kwargs
        tex_text = TexText(tex_string, isolate=labelled_content, **kwas1)
        lines = (tex_text.select_part(part) for part in labelled_content)

        kwas2 = default_kwargs_vmobj | kwargs
        super().__init__(*lines, **kwas2)

        for i in range(len(labelled_content)):
            self[i].shift(DOWN*buff*i)
        #self.arrange(DOWN, center=False, buff=buff, aligned_edge=aligned_edge)
        for i in range(len(items)):
            self[i].shift(RIGHT*DEFAULT_MOBJECT_TO_EDGE_BUFF*items[i][0])

    def fade_all_but(self, index: int, opacity: float = 0.25) -> None:
        for i, part in enumerate(self.submobjects):
            part.set_fill(opacity=(1.0 if i == index else opacity))

class Integer(Integer):
    def __init__(self, number = 0, num_decimal_places = 0, **kwargs):
        kwargs = default_kwargs_vmobj | kwargs
        super().__init__(number, num_decimal_places, **kwargs)

class DecimalNumber(DecimalNumber):
    def __init__(self, number = 0, color = FRONTCOL, stroke_width = 0, fill_opacity = 1, fill_border_width = 0.5, num_decimal_places = 2, include_sign = False, group_with_commas = True, digit_buff_per_font_unit = 0.001, show_ellipsis = False, unit = None, include_background_rectangle = False, edge_to_fix = LEFT, font_size = CONTENT_FONT_SIZE, text_config = dict(), **kwargs):
        kwargs = default_kwargs_vmobj | kwargs
        super().__init__(number, color, stroke_width, fill_opacity, fill_border_width, num_decimal_places, include_sign, group_with_commas, digit_buff_per_font_unit, show_ellipsis, unit, include_background_rectangle, edge_to_fix, font_size, text_config, **kwargs)

class Write(Write):
    def __init__(self, vmobject, run_time = 0.5, lag_ratio = -1, rate_func = linear, stroke_color = None, **kwargs):
        super().__init__(vmobject, run_time, lag_ratio, rate_func, stroke_color, **kwargs)

class LenseThirringGL(Slide):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.high_quality = False
        kwargs['show_animation_progress'] = True
        #kwargs['leave_progress_bars'] = True
        kwargs['camera_config'] = {'background_color':BACKCOL}
        kwargs['camera_config']['light_source_position'] = np.array([10, -10, 10])
        #kwargs['start_at_animation_number'] = 7
        #kwargs['end_at_animation_number'] = 18
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
        #ltt.set_params_lense_thirring(mass=6.96e-10,omega=2.64e-5,radius=1)
        #   somewhat unrealistic parameters
        ltt.set_params_lense_thirring(mass=1.0,omega=1.0,radius=1.0)
        sphere_omega0 = 2*np.pi
        sphere_omega = sphere_omega0
        #   fix_in_frame objs z_index
        self.z_idx_fix = default_kwargs_text['z_index']
        #   set 2d screen space offset of 3d objects
        self.camera.uniforms['shift_screen_space'] = np.array([2.2,0.0])
        self.offset_3d = ComplexValueTracker(self.camera.uniforms['shift_screen_space'][0]+1j*self.camera.uniforms['shift_screen_space'][1])
        def screen_space_shift(obj,dt):
            nonlocal self
            val = self.offset_3d.get_value()
            self.camera.uniforms['shift_screen_space'] = np.array([val.real,val.imag],dtype=float)
        self.offset_3d.add_updater(screen_space_shift)
        #   predefined colors for certain tex parts
        symCols = {
            r'\vec{ x }':BLUE_D,
            r'{x}':BLUE_D,
            r'{g}':YELLOW_D,
            r'\bm{g}':YELLOW_D,
            r'{h}':YELLOW_B,
            r'{u}':GREEN_C,
            r'{v}':LIGHT_BROWN,
            r'\Gamma':PURPLE_C,
            r'{ R }':GREEN_C,
            r'{R}':LIGHT_BROWN,
            r'{T}':RED,
            r'\vec{ F }':GOLD_C,
            r'\vec{ F }_E':GOLD_C,
            r'\vec{ F }_B':GOLD_C,
            r'\vec{ E }':YELLOW_B,
            r'\vec{ B }':YELLOW_B,
            r'{B}':YELLOW_B,
            r'\vec{ v }':BLUE_B,
            r'{m}':WHITE,
            r'\vec{ L }':BLUE_D,
            r'\vec{ r }':GREEN_C,
            r'\vec{ \Omega }':ORANGE,
            r'\vec{ M }': LIGHT_BROWN
        }
        #   function to align Mobjs on the left below the title
        def align_mobjs(mobjs,tomobj,center=False):
            for i,mobjl in enumerate(mobjs):
                last = mobjs[i-1][0] if i>0 else tomobj
                for mobj in mobjl:
                    mobj.fix_in_frame()
                    if not center:
                        mobj.next_to(last.get_corner(DL),DOWN,aligned_edge=LEFT)
                    else:
                        mobj.next_to(last.get_corner(DOWN),DOWN,aligned_edge=ORIGIN)
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
        self.canvas_objs.append(self.slide_title)

        presentation_info = TexText(PresentationInfo, alignment=None)
        presentation_info.next_to(TextBox.get_corner(DL), RIGHT+UP, MED_LARGE_BUFF)

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
        self.canvas_objs.append(contact_info)
        #   Präsentationstitel
        presentation_title = TexText(PresentationTitle, font_size=CONTENT_FONT_SIZE*0.6).align_to(contact_info, UP)
        self.canvas_objs.append(presentation_title)
        #   Seitennummer
        self.slide_number_val = 1
        self.slide_number = Integer(1, font_size=CONTENT_FONT_SIZE*0.6, z_index=self.z_idx_fix).to_corner(DR,buff=MED_SMALL_BUFF)
        self.slide_number.fix_in_frame()
        self.canvas_objs.append(self.slide_number)
        self.play(FadeOut(presentation_info),FadeOut(TextBox),FadeOut(background_render), Write(contact_info), Write(presentation_title), Write(self.slide_number), self.slide_title.animate.to_corner(UL))
        self.pause()


        # Inhaltsverzeichnis (2-6)
        Inhalt = BulletedList(*PresentationContents,buff=0)
        Inhalt.to_edge(LEFT)
        Inhalt.fix_in_frame()
        indended = []
        for istr in range(len(PresentationContents)):
            if len(indended)>0 and PresentationContents[istr][0] == 0:
                self.play(*indended)
                self.pause()
                indended = []
            indended.append(Write(Inhalt[istr]))
        self.play(*indended)
        self.pause()


        minkmetric = TexText(r'$\bm{\eta}=\begin{bmatrix}1 & 0 & 0 & 0 \\0 & -1 & 0 & 0 \\0 & 0 & -1 & 0 \\0 & 0 & 0 & -1\end{bmatrix}$')
        minkmetric_conv = TexText(r'$(+,-,-,-)$')
        natEinh = TexText(r'$c=G=1$')
        minkmetric.move_to((FRAME_WIDTH/4,0,0))
        self.play(Write(minkmetric))
        self.pause()


        natEinh.next_to(ude_logo.get_corner(UL),LEFT,aligned_edge=TOP)
        minkmetric_conv.next_to(natEinh.get_corner(DR),DOWN,aligned_edge=RIGHT)
        self.play(ReplacementTransform(minkmetric,minkmetric_conv),Write(natEinh))
        self.canvas_objs.append(minkmetric_conv)
        self.canvas_objs.append(natEinh)
        back_rects = []
        back_rects_vmobj = None
        def update_back_rects():
            nonlocal back_rects,back_rects_vmobj
            if len(back_rects) > 0:
                self.canvas_objs.remove(back_rects_vmobj)
                self.remove(back_rects_vmobj)
                back_rects = []
            for i,obj in enumerate([self.slide_title,minkmetric_conv,natEinh,ude_logo,contact_info,presentation_title,self.slide_number]):
                rect = SurroundingRectangle(obj, color=BACKCOL, fill_opacity=0.6, stroke_width=0,z_index=self.z_idx_fix-1)
                rect.fix_in_frame()
                back_rects.append(rect)
            back_rects_vmobj = VGroup(*back_rects)
            self.add(back_rects_vmobj)
            self.canvas_objs.append(back_rects_vmobj)
        update_back_rects()
        self.pause()


        # ART (7-17)
        self.setup_new_slide(title="Allgemeine Relativitätstheorie", cleanup=True)
        update_back_rects()
        self.pause()


        texts = [
            (0, 'entwickelt von Albert Einstein'),
            (0, 'beruht auf der Differentialgeometrie'),
            (0, 'angewendet bei GPS'),
            (0, r'beschreibt Gravitation nicht als Kraft $F_{\mathrm{G}}$ sondern \\als Effekt einer vierdimensionalen Raumzeit'),
            (0, r'Raumzeit ist pseudo-riemannsche \\Mannigfaltigkeit mit Linienelement:')
        ]
        Texts = BulletedList(*texts)
        Texts.next_to(self.slide_title, direction=DOWN, aligned_edge=LEFT)
        Texts.fix_in_frame()
        line_el = TexText(r'$\mathrm{d}s^2 = {g}_{\mu\nu}\mathrm{d}{x}^{\mu}\mathrm{d}{x}^{\nu}$', isolate=[r'{g}',r'{x}']).fix_in_frame().set_color_by_tex_to_color_map(symCols)
        line_el.next_to(Texts[-1],direction=DOWN,aligned_edge=LEFT).shift(RIGHT*DEFAULT_MOBJECT_TO_EDGE_BUFF)
        for text in Texts:
            self.play(Write(text))
            self.pause()
        self.play(Write(line_el))
        self.pause()


        basesize = 5
        basepos = RIGHT*4+DOWN*1.5
        grid_img = ImageMobject('./assets/grid_skizze.png', height=basesize*1.324503311,z_index=-2).fix_in_frame().shift(basepos)
        grid_back = ImageMobject('./assets/grid_back.png', height=basesize*1.59602649,z_index=-1).fix_in_frame().shift(basepos)
        back_f = ImageMobject('./assets/Kraft.png', height=basesize,z_index=0).fix_in_frame().shift(basepos)
        back_r = ImageMobject('./assets/Raumzeit.png', height=basesize,z_index=0).fix_in_frame().shift(basepos)
        apple_f = ImageMobject('./assets/apple_force.png', height=basesize*0.179470199,z_index=1).fix_in_frame().shift(basepos)
        apple = ImageMobject('./assets/apple.png', height=basesize*0.179470199,z_index=1).fix_in_frame().shift(basepos)
        self.add(grid_back)
        self.play(FadeIn(back_f),FadeIn(apple_f))
        self.pause(loop=True)


        self.play(apple_f.animate.shift(1.2*DOWN),run_time=2.0,rate_func=lambda t: t**2)
        apple_f.shift(1.2*UP)
        self.pause()

        self.add(back_r, apple)
        self.play(FadeOut(apple_f),FadeOut(back_f),FadeIn(grid_img))
        self.pause(loop=True)


        self.play(apple.animate.shift(1.2*DOWN),grid_img.animate.shift(1.2*DOWN),run_time=2.0,rate_func=lambda t: t**2)
        apple.shift(1.2*UP)
        grid_img.shift(1.2*UP)
        self.pause()


        # Metrik & Geodätengleichung (18-26)
        self.setup_new_slide(title='Metrik und Geodäten', cleanup=True)
        update_back_rects()
        self.pause()


        #   Euklidisch (19-22)
        fläche_text = TexText(r'Fläche')
        rfunc = TexText(r'$\vec{ x } : \mathbb{R}^2 \rightarrow \mathbb{R}^3$',isolate=[r'\vec{ x }']).set_color_by_tex_to_color_map(symCols)
        rfunc_euclid = TexText(r'$\vec{ x }({u},{v}) = ({u},{v},0)$',isolate=[r'{u}',r'{v}',r'\vec{ x }']).set_color_by_tex_to_color_map(symCols)
        metrik_text = TexText(r'Metrik')
        g_tensor = TexText(r'${g}_{\mu \nu} = \partial_{\mu} \vec{ x }\cdot\partial_{\nu} \vec{ x }$', isolate=[r'{g}', r'\vec{ x }']).set_color_by_tex_to_color_map(symCols)
        g_euclid = VGroup(TexText(r'$\bm{g} = $',isolate=r'\bm{g}').set_color_by_tex_to_color_map(symCols,only_isolated=True),TexText(r'$\left[\begin{array}{c}1 \quad 0 \\0 \quad 1 \\\end{array}\right]$')).arrange()
        geod_gl_text = TexText(r'Geodätengleichung')
        geod_gl_t1 = TexText(r'$\frac{\mathrm{d}^2 {x}^{\lambda}}{\mathrm{d} \tau^2} = $',isolate=[r'{x}']).set_color_by_tex_to_color_map(symCols)
        geod_gl_t2 = TexText(r'$-\Gamma^{\lambda}_{\mu \nu}[\bm{g}(\vec{ x })] \frac{\mathrm{d} {x}^{\mu}}{\mathrm{d} \tau} \frac{\mathrm{d} {x}^{\nu}}{\mathrm{d} \tau}$',isolate=[r'{x}',r'\bm{g}',r'\vec{ x }',r'\Gamma']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        geod_gl_eucild = VGroup(geod_gl_t1.copy(),TexText(r'$0$')).arrange()
        geod_gl = VGroup(geod_gl_t1,geod_gl_t2).arrange()
        itms = [(fläche_text,),(rfunc,rfunc_euclid),(metrik_text,),(g_tensor,),(g_euclid,),(geod_gl_text,),(geod_gl_eucild,geod_gl)]
        align_mobjs(itms, self.slide_title)

        ltt.set_params_gaußian_surface(1/np.sqrt(2),A=0.0)
        grid_nc,_ = mt.get_grid_surface(uv_func=lambda u,v: [u,v,v*0], u_range=(-3,3), v_range=(-3,3), grid_size=(8,8), grid_col=FRONTCOL)
        ts,zs = ltt.get_geodesic(6, np.array([3,0.5,-1,0]),tol=1e-9,accF=ltt.acc_gaußian_surface,check_break=lambda t,r: np.abs(r[0])>3 or np.abs(r[1])>3)
        pcd_nc = mt.CurveDrawer([ltt.ParametricCurve(ts,[[uv[0],uv[1],0] for uv in zs[:,:2]])],fixed_color=ORANGE)
        pcd_nc.update_graphics()
        dot = Sphere(radius=0.05,color=symCols[r'\vec{ x }'],shading=(0,0,0)).apply_depth_test()
        u_label = TexText(r'${u}$', fix_in_frame=False).next_to(grid_nc.get_bottom(),DOWN).set_color_by_tex_to_color_map(symCols)
        v_label = TexText(r'${v}$', fix_in_frame=False).next_to(grid_nc.get_left(),LEFT).set_color_by_tex_to_color_map(symCols)
        self.play(Write(grid_nc), Write(pcd_nc), Write(u_label), Write(v_label))
        self.pause()


        self.play(Write(fläche_text),Write(rfunc_euclid))
        self.pause()


        self.play(Write(metrik_text),Write(g_tensor), Write(g_euclid))
        self.pause(auto_next=True)


        self.play(Write(geod_gl_text),Write(geod_gl_eucild))
        self.pause(loop=True)


        #   Animation euklidischer Raum (23)
        self.add(dot)
        #camRot.startUpdating()
        self.play(MoveAlongPath(dot,pcd_nc), run_time=4.0,rate_func=linear)
        #camRot.stopUpdating()
        self.remove(dot)
        self.pause(auto_next=True)


        #   Gekrümmter Raum (24-26)
        ltt.set_params_gaußian_surface(1/np.sqrt(2),A=2.0)
        _, surface = mt.get_grid_surface(uv_func=lambda u,v: [u,v,ltt.z_gaußian_surface(u,v)], u_range=(-3,3), v_range=(-3,3), grid_size=(8,8), grid_col=FRONTCOL)
        surface.set_color_by_rgba_func(lambda r: get_color_map('viridis')(r[2]/ltt.z_gaußian_surface(0,0)))
        ts,zs = ltt.get_geodesic(10, np.array([3,0.5,-1,0]),tol=1e-9,accF=ltt.acc_gaußian_surface,check_break=lambda t,r: np.abs(r[0])>3 or np.abs(r[1])>3)
        pcd = mt.CurveDrawer([ltt.ParametricCurve(ts,[[uv[0],uv[1],ltt.z_gaußian_surface(uv[0],uv[1])+1e-2] for uv in zs[:,:2]])],fixed_color=ORANGE)
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
        self.play(self.next_slide_number_animation(), FadeIn(surface), grid_nc.animate.apply_function(lambda r: [r[0],r[1],ltt.z_gaußian_surface(r[0],r[1])+1e-2]), ReplacementTransform(pcd_nc, pcd), ReplacementTransform(rfunc_euclid,rfunc), ReplacementTransform(g_euclid[1],metric), ReplacementTransform(geod_gl_eucild[1],geod_gl[1]))
        self.wait(0.04)
        self.pause(loop=True)


        metric.add_updater(m_update)
        self.add(dot)
        #camRot.startUpdating()
        self.play(MoveAlongPath(dot,pcd), run_time=4.0,rate_func=linear)
        #camRot.stopUpdating()
        self.pause()


        # EFGl mit Analogie 2D Fläche eingebettet in 3D Raum -> 4D Fläche (27-33)
        self.setup_new_slide(title='Einsteinsche Feldgleichungen',cleanup=True)
        update_back_rects()
        text1 = TexText(r'2D Fläche $\rightarrow$ 4D Mannigfaltigkeit')
        text2 = TexText(r'Koordinaten $(ct,x,y,z)$ $\Rightarrow$ $\bm{g}\in\mathbb{R}^{4\times 4}$',isolate=[r'\bm{g}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        efe = TexText(r'${ R }_{\mu \nu} - \frac{1}{2} {g}_{\mu \nu} {R} = 8 \pi{T}_{\mu \nu}$',isolate=[r'{g}',r'\bm{g}',r'{ R }',r'{R}', r'{T}'],font_size=CONTENT_FONT_SIZE*1.5).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        RicciT = TexText(r'Ricci-Tensor: ',r'${ R }_{\mu \nu}\left[\bm{g}\right]$',isolate=[r'{g}',r'\bm{g}',r'{ R }',r'{R}', r'{T}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        KrmmS = TexText(r'Krümmungsskalar: ',r'${R}\left[\bm{g}\right]$',isolate=[r'{g}',r'\bm{g}',r'{ R }',r'{R}', r'{T}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        EnImpT = TexText(r'Energie-Impuls-Tensor: ',r'${T}_{\mu \nu}$',isolate=[r'{g}',r'\bm{g}',r'{ R }',r'{R}', r'{T}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        itms = [(text1,),(text2,),(efe,),(RicciT,),(KrmmS,),(EnImpT,)]
        align_mobjs(itms[:2],self.slide_title)
        efe.center()
        align_mobjs(itms[3:],efe,center=True)
        for itm in itms:
            self.pause()
            self.play(Write(itm[0]))
        self.pause()
        

        # Linearisierung & Gravitoelektromagnetismus (34-40)
        self.setup_new_slide(title='Linearisierung',cleanup=True)
        update_back_rects()
        geod_gl = TexText(r'$\frac{\mathrm{d}^2 {x}^{\lambda}}{\mathrm{d} \tau^2} = -\Gamma^{\lambda}_{\mu \nu}[\bm{g}(\vec{ x })] \frac{\mathrm{d} {x}^{\mu}}{\mathrm{d} \tau} \frac{\mathrm{d} {x}^{\nu}}{\mathrm{d} \tau}$',isolate=[r'{x}',r'\bm{g}',r'\vec{ x }',r'\Gamma']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe = TexText(r'${ R }_{\mu \nu} - \frac{1}{2} {g}_{\mu \nu} {R} = 8 \pi {T}_{\mu \nu}$',isolate=[r'{g}',r'\bm{g}',r'{ R }',r'{R}', r'{T}']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        approxs = TexText(r'Annahmen: ', r'${g}_{\mu \nu} = \eta_{\mu \nu} + {h}_{\mu \nu}$, $\bm{h} \ll \bm{\eta}$, $\tau \approx t$',isolate=[r'{g}',r'\bm{g}',r'{h}',r'\bm{h}']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        fields = TexText(r'Substitutionen: ',r'$\vec{ E }=\frac{1}{2}\vec{\nabla} {h}_{00}$, ${B}_j=-\varepsilon_{jlm}\frac{\partial {h}_{0m}}{\partial {x}^l}$',isolate=[r'\vec{ E }',r'{h}',r'{x}',r'{B}']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        geod_gl_l1 = TexText(r'$\frac{d^2 {x}^{i}}{{dt}^2} = -\frac{1}{2}\frac{\partial {h}_{00}}{\partial {x}^i} + \varepsilon_{ijk}\varepsilon_{jlm}\frac{\partial {h}_{0m}}{\partial {x}^l}\frac{d {x}^k}{dt}$',isolate=[r'{x}',r'\bm{g}', r'{h}']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        geod_gl_l2 = TexText(r'$\vec{ F } = {m} \left( \vec{ E } + \vec{ v }\times\vec{ B } \right)$',isolate=[r'\vec{ F }', r'{m}',r'\vec{ E }', r'\vec{ v }', r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe_l1 = TexText(r'$-\Delta {h}_{00} = 8\pi\rho$, $-\Delta {h}_{0i} = 16\pi j_i$',isolate=[r'{h}']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe_l2 = TexText(r'$\vec{\nabla}\cdot\vec{ E } = - 4 \pi \rho$',isolate=[r'\vec{ E }', r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe_l3 = TexText(r'$\vec{\nabla}\cdot\vec{ B } = 0$',isolate=[r'\vec{ E }', r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe_l4 = TexText(r'$\vec{\nabla}\times\vec{ E } = 0$',isolate=[r'\vec{ E }', r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        efe_l5 = TexText(r'$\vec{\nabla}\times\vec{ B } = -16 \pi \vec{j}$',isolate=[r'\vec{ E }', r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True).fix_in_frame()
        align_mobjs([(approxs,),(fields,),(efe,)],self.slide_title)
        efe.shift(DOWN*MED_SMALL_BUFF)
        align_mobjs([(efe_l1,),(efe_l2,)],efe)
        efe_l2.shift(DOWN*MED_SMALL_BUFF)
        align_mobjs([(efe_l3,),(efe_l4,),(efe_l5,)],efe_l2)
        geod_gl.next_to((0,efe.get_edge_center(TOP)[1],0),RIGHT,aligned_edge=UP)
        align_mobjs([(geod_gl_l1,),(geod_gl_l2,)],geod_gl)
        geod_gl_l2.shift(DOWN*MED_LARGE_BUFF*2)
        self.pause()


        self.play(Write(approxs))
        self.pause()


        self.play(Write(efe),Write(geod_gl))
        self.pause()


        self.play(Write(efe_l1))
        self.pause()


        self.play(Write(fields),Write(efe_l2),Write(efe_l3),Write(efe_l4),Write(efe_l5),self.next_slide_title_animation('Gravitoelektromagnetismus'))
        self.pause()


        self.play(Write(geod_gl_l1))
        self.pause()


        self.play(Write(geod_gl_l2))
        self.pause()


        # Rotierende Kugelmasse (41-46)
        self.setup_new_slide(title='Rotierende Kugelmasse', cleanup=True)
        update_back_rects()
        rhoKug = TexText(r'$\rho(|\vec{ x }|) = \rho_0 \Theta(R-|\vec{ x }|)$',isolate=[r'\vec{ x }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        jKug = TexText(r'$\vec{j} = \rho_0 \vec{\omega}\times\vec{ x }\Theta(R-|\vec{ x }|)$',isolate=[r'\vec{ x }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        bfield_formula = TexText(r'$\vec{ B } = \frac{2}{|\vec{ x }|^3} \left[\vec{S} - \frac{3 (\vec{S}\cdot\vec{ x }) \vec{ x }}{|\vec{ x }|^2}\right]$',isolate=[r'\vec{ B }',r'\vec{ x }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        efield_formula = TexText(r'$\vec{ E } = -\frac{M \vec{ x }}{|\vec{ x }|^3}$',isolate=[r'\vec{ E }',r'\vec{ x }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        inertia = TexText(r'$I = \frac{2}{5} M R^2$')
        angularM = TexText(r'$\vec{S} = I \vec{\omega}$')
        align_mobjs([(rhoKug,),(jKug,),(efield_formula,),(bfield_formula,),(inertia,),(angularM,)],self.slide_title)
        self.pause()


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
        self.canvas_objs.append(sphere)
        self.play(Write(axes),ShowCreation(sphere),Write(rhoKug))
        self.pause()


        newnatEinh = TexText(r'$c=G=R=1$').next_to(ude_logo.get_corner(UL),LEFT,aligned_edge=TOP)
        self.play(ReplacementTransform(natEinh,newnatEinh))
        self.canvas_objs.append(newnatEinh)
        self.canvas_objs.remove(natEinh)
        natEinh = newnatEinh
        update_back_rects()
        self.pause(auto_next=True)


        self.play(Write(jKug))
        self.pause(loop=True)


        sphere.add_updater(rotater)
        self.wait(4)
        self.pause(auto_next=True)


        self.play(Write(efield_formula), Write(bfield_formula), Write(inertia), Write(angularM))
        self.pause(auto_next=True)


        # EM-Felder (47-52)
        self.play(self.next_slide_title_animation('EM-Felder'))
        update_back_rects()
        self.pause(loop=True)


        self.wait(4)
        self.pause(auto_next=True)
        

        formula_box = SurroundingRectangle(efield_formula, color=ORANGE)
        bfmax = np.linalg.norm(ltt.bfield(np.array([[0,0,ltt.R]])))
        efmax = np.linalg.norm(ltt.efield(np.array([[0,0,ltt.R]])))
        bounds = np.array([[-2.5,-2.5,-2.5],[2.5,2.5,2.5]])

        #   E-Feld
        sls_e = mt.StreamLines(fieldf=lambda t,x: ltt.efield(np.array([x])), boundary=bounds, system_timescale=1/efmax, vmax=efmax)
        self.play(self.next_slide_number_animation(),Write(formula_box),Write(axes),Write(sls_e))
        self.pause(loop=True)

        
        #   E-Feld Animation
        sls_e.startUpdating(timeScaleF=0.25)
        camRot.startUpdating()
        self.wait(4.0)
        self.pause(auto_next=True)


        #   B-Feld
        sls_e.stopUpdating()
        camRot.stopUpdating()
        rs = [1, 2, 1]
        zs = [-2, 0, 2]
        phis = np.arange(0,2*np.pi,np.pi/4)
        startPoints = np.array([[rs[zi]*np.cos(phi),rs[zi]*np.sin(phi),zs[zi]] for phi in phis for zi in range(len(zs))])
        sls_b = mt.StreamLines(fieldf=lambda t,x: ltt.bfield(np.array([x])), startPoints=startPoints, boundary=bounds, system_timescale=1/bfmax, vmax=bfmax)
        self.play(self.next_slide_number_animation(),formula_box.animate.surround(bfield_formula), FadeOut(sls_e), Write(sls_b))
        self.pause(loop=True)


        #   B-Feld Animation
        sls_b.startUpdating(timeScaleF=0.25)
        camRot.startUpdating()
        self.wait(4.0)
        self.pause()


        # Trajektorien (53-64)
        self.remove(sls_e)
        self.remove(sls_b)
        camRot.stopUpdating()
        sphere_omega = 0.0
        camRot.update(camRot.frame, 0)
        self.setup_new_slide(title=r'Trajektorien',cleanup=True)
        update_back_rects()
        self.pause()


        force = TexText(r'$\vec{ F } = {m} \left( \vec{ E } + \vec{ v }\times\vec{ B } \right)$',isolate=[r'\vec{ F }',r'\vec{ v }',r'\vec{ E }',r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        omega_text = TexText(r'$\omega = $')
        omega_val = DecimalNumber(0.0, num_decimal_places=2).fix_in_frame()
        align_mobjs([(force,),(omega_text,)],self.slide_title)
        omega_val.next_to(omega_text,RIGHT)
        self.play(Write(force),Write(omega_text), Write(omega_val), self.offset_3d.animate.set_value(0+0j), sphere.animate.scale(0.1))
        #sphere.deactivate_depth_test()
        #sphere.set_z_index(-10)
        self.pause()

        traj_tol = 1e-8
        x0_v0_omega = [
            ([3,0.5,0],[-0.6,0,0],[0.0,1.0],[10,10]),
            ([0,0,2],[-np.sqrt(0.5),0,0],[0.0,0.712],[18,93.5]),
        ]
        trajs = [
            mt.CurveDrawer([ltt.get_trajectory(M=ltt.M,R=ltt.R,omega=x0_v0_omega[0][2][0],x0=x0_v0_omega[0][0],v0=x0_v0_omega[0][1],tmax=x0_v0_omega[0][3][0],cputmax=2,tol=traj_tol)],randomize_t0s=False),
            mt.CurveDrawer([ltt.get_trajectory(M=ltt.M,R=ltt.R,omega=x0_v0_omega[1][2][0],x0=x0_v0_omega[1][0],v0=x0_v0_omega[1][1],tmax=x0_v0_omega[1][3][0],cputmax=2,tol=traj_tol)],randomize_t0s=False),
        ]
        current_traj = 0
        omega_tracker = ValueTracker(0.0)
        tmax_tracker = ValueTracker(0.0)
        def traj_updater(obj,dt):
            nonlocal trajs, current_traj, x0_v0_omega, omega_tracker, tmax_tracker, sphere_omega
            omega = omega_tracker.get_value()
            omega_val.set_value(omega)
            traj = trajs[current_traj]
            traj.set_pcs([ltt.get_trajectory(M=ltt.M,R=ltt.R,omega=omega,x0=x0_v0_omega[current_traj][0],v0=x0_v0_omega[current_traj][1],tmax=tmax_tracker.get_value(),cputmax=2,tol=traj_tol)])
            traj.update_graphics()
            sphere_omega = omega*sphere_omega0
        omega_tracker.add_updater(traj_updater)

        tmax_tracker.set_value(x0_v0_omega[0][3][0])
        traj_updater(omega_tracker,0)
        self.play(Write(trajs[0],run_time=2.0),run_time=2.0,rate_func=linear)
        self.pause(auto_next=True)


        self.play(omega_tracker.animate.set_value(x0_v0_omega[0][2][1]),run_time=4.0,rate_func=linear)
        self.pause(loop=True)


        self.wait(2*np.pi/sphere_omega)
        self.pause()

        
        omega_tracker.remove_updater(traj_updater)
        current_traj = 1
        omega_tracker.set_value(x0_v0_omega[1][2][0])
        tmax_tracker.set_value(x0_v0_omega[1][3][0])
        traj_updater(omega_tracker,0)
        self.play(self.next_slide_number_animation(),FadeOut(trajs[0]),sphere.animate.scale(10))
        self.play(Write(trajs[1],run_time=2.0),run_time=2.0,rate_func=linear)
        omega_tracker.add_updater(traj_updater)
        self.pause(auto_next=True)
        

        self.play(omega_tracker.animate.set_value(x0_v0_omega[1][2][1]),run_time=4.0,rate_func=linear)
        omega_tracker.remove_updater(traj_updater)
        self.pause(loop=True)


        self.wait(2*np.pi/sphere_omega)
        self.pause(auto_next=True)

        
        omega_tracker.add_updater(traj_updater)
        self.play(tmax_tracker.animate.set_value(x0_v0_omega[1][3][1]/2),self.frame.animate.reorient(0,58,0),run_time=2.0,rate_func=linear)
        self.play(tmax_tracker.animate.set_value(x0_v0_omega[1][3][1]),self.frame.animate.reorient(0,0,0),run_time=2.0,rate_func=linear)
        self.pause(loop=True)


        self.wait(2*np.pi/sphere_omega)
        omega_tracker.remove_updater(traj_updater)
        self.pause()


        #   Raumzeitdarstellung (65-70)
        sphere_omega = 0.0
        self.setup_new_slide(title='Raumzeitdarstellung',cleanup=True)
        update_back_rects()
        self.pause()


        text_erkl0 = TexText(r'Zeitentwicklung')
        text_erkl1 = TexText(r'für jeden')
        text_erkl2 = TexText(r'Gitterpunkt')
        text_o0 = TexText('bei ', r'$\omega=0$')
        text_o1 = TexText('bei ', r'$\omega=1$')
        align_mobjs([(text_erkl0,),(text_erkl1,),(text_erkl2,),(text_o0,text_o1)],self.slide_title)
        #sphere.apply_depth_test()
        self.play(Write(text_erkl0), Write(text_erkl1), Write(text_erkl2))

        lines_0 = np.load('./assets/spacetime_sims/lt2d_lines__line_nums=22__subdivisions=100__timesteps=180__tau_max=9.0__R=1.0__M=1.0__omega=0.0.npy')
        print(f'loaded lines with shape {lines_0.shape}')
        lines_1 = np.load('./assets/spacetime_sims/lt2d_lines__line_nums=22__subdivisions=100__timesteps=180__tau_max=9.0__R=1.0__M=1.0__omega=1.0.npy')
        print(f'loaded lines with shape {lines_1.shape}')
        lanim_0 = mt.LineAnim(np.linspace(0,9.0,lines_0.shape[1]),lines_0,z_index=-1)
        lanim_1 = mt.LineAnim(np.linspace(0,9.0,lines_1.shape[1]),lines_1,z_index=-1)
        lanim_0.updateVMobjs(0,force=True)
        lanim_1.updateVMobjs(0,force=True)
        # overlap square from (-20,-20,0.1) to (20,20,0.1) except for the region between (-3,-3,0.1) and (3,3,0.1)
        olaps = 3
        verts = (
            (-20,-20,0.1),
            (20,-20,0.1),
            (20,20,0.1),
            (-20,20,0.1),
            (-20,-20,0.1),
            (-olaps,-olaps,0.1),
            (-olaps,olaps,0.1),
            (olaps,olaps,0.1),
            (olaps,-olaps,0.1),
            (-olaps,-olaps,0.1),
        )
        overlap = Polygon(*verts, fill_color=BACKCOL, fill_opacity=1.0, stroke_width=0.0).apply_depth_test()
        olapedge = Polygon(*verts[5:-1], fill_opacity=0.0, stroke_width=DEFAULT_STROKE_WIDTH, stroke_color=FRONTCOL)
        self.add(overlap,olapedge)
        self.play(Write(text_o0), Write(lanim_0))
        self.pause(loop=True)


        lanim_0.startUpdating(timeScaleF=2.0)
        self.wait(4.0)
        lanim_0.stopUpdating()
        self.pause()

        
        self.play(self.next_slide_number_animation(),ReplacementTransform(text_o0,text_o1), ReplacementTransform(lanim_0, lanim_1))
        sphere_omega = sphere_omega0
        self.pause(loop=True)


        lanim_1.startUpdating(timeScaleF=2.0)
        self.wait(4.0)
        lanim_1.stopUpdating()
        self.pause(auto_next=True)


        #  Präzession Intuition (71-78)
        lanim_1.startUpdating(timeScaleF=2.0)
        self.play(self.next_slide_number_animation(), self.next_slide_title_animation('Präzession'), self.wipe([text_erkl0,text_erkl1,text_erkl2,text_o1],[],return_animation=True))
        update_back_rects()
        self.wait(4.0-DEFAULT_ANIMATION_RUN_TIME)
        lanim_1.stopUpdating()
        self.pause()


        probe = TexturedSurface(Sphere(radius=0.2), "./assets/grid.png", depth_test=True)
        probe.move_to((2.0,0,0))
        precession_angle = -np.pi/8
        probe.rotate(precession_angle,axis=UP)
        spin_axis = rotate_vector((0,0,1), precession_angle, axis=UP)
        spin = 0
        precession = 0
        def rot_around_spin(obj,dt):
            nonlocal spin_axis, spin
            obj.rotate(dt*spin,axis=spin_axis)
        probe.add_updater(rot_around_spin)
        probe_arr = mt.Arrow3D(start=(2,0,0),end=(2,0,0.5),tip_width_ratio = 0.3,tip_length = 0.1,shaft_width = 0.015,color=BLUE_D,depth_test=True)
        probe_arr.rotate(precession_angle,axis=UP,about_point=(2,0,0))
        def rot_spin(obj,dt):
            nonlocal spin_axis, probe, precession
            probe.rotate(dt*precession,axis=(0,0,-1),about_point=(2,0,0))
            probe_arr.rotate(dt*precession,axis=(0,0,-1),about_point=(2,0,0))
            spin_axis = rotate_vector(spin_axis, dt*precession, axis=(0,0,-1))
        probe_arr.add_updater(rot_spin)

        precession_arrow = mt.Arrow3D(start=(2,0,0),end=(2,0,0.462),tip_width_ratio = 0.3,tip_length = 0.1,shaft_width = 0.015,color=ORANGE,depth_test=True)
        precession_circle = Circle(radius=0.191, color=YELLOW_E, fill_opacity=0.4, stroke_width=DEFAULT_STROKE_WIDTH).rotate(np.pi,axis=UP)
        precession_circle.move_to((2.0,0,0.462))

        lines_2 = np.load('./assets/spacetime_sims/lt2d_lines__line_nums=20__subdivisions=100__timesteps=180__tau_max=9.0__R=1.0__M=1.0__omega=1.0.npy')
        print(f'loaded lines with shape {lines_2.shape}')
        lanim_2 = mt.LineAnim(np.linspace(0,9.0,lines_2.shape[1]),lines_2,z_index=-1)
        lanim_2.updateVMobjs(0,force=True)

        sphere_omega = 0.0
        lanim_1.reset_state()
        self.play(self.frame.animate.reorient(75,40,0,(2,0,0),3.0),rate_func=linear)
        self.remove(overlap,olapedge)
        self.play(ShowCreation(probe),ShowCreation(probe_arr),Write(lanim_2))
        self.pause(loop=True)


        sphere_omega = sphere_omega0
        precession = np.pi/2
        spin = 5
        lanim_1.startUpdating(timeScaleF=2.0)
        lanim_2.startUpdating(timeScaleF=2.0)
        self.wait(4.0)
        lanim_1.stopUpdating()
        lanim_2.stopUpdating()
        self.pause(auto_next=True)


        lanim_1.startUpdating(timeScaleF=2.0)
        lanim_2.startUpdating(timeScaleF=2.0)
        self.play(ShowCreation(precession_circle),ShowCreation(precession_arrow),run_time=1,rate_func=linear)
        self.wait(3.0)
        lanim_1.stopUpdating()
        lanim_2.stopUpdating()
        self.pause(loop=True)


        lanim_1.startUpdating(timeScaleF=2.0)
        lanim_2.startUpdating(timeScaleF=2.0)
        self.wait(4.0)
        lanim_1.stopUpdating()
        lanim_2.stopUpdating()
        self.pause()


        #   Präzession Rechnung (79-90)
        # slidenumber, 3d view nach rechts, kreisel in (0,0,0), Erde & Linien & Kreis & pr_vec weg 
        drehimp = TexText('Drehimpuls:')
        drehimp_tex = TexText(r'$\vec{ L }$',isolate=[r'\vec{ L }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        magn = TexText('Gravitomagnetisches Feld:')
        magn_tex = TexText(r'$\vec{ B } = \frac{2 \vec{ S }}{{r}^3}$',isolate=[r'\vec{ B }',r'\vec{ S }',r'{r}']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        lforce = TexText('Lorentz-Kraft:')
        lforce_tex = TexText(r'$\vec{ F } = m \vec{ v }\times\vec{ B }$',isolate=[r'\vec{ F }',r'\vec{ v }',r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        torque = TexText('Drehmoment:')
        torque_tex = TexText(r'$\frac{\mathrm{d} \vec{ L }}{\mathrm{d} t} = \vec{ M } = \vec{ r }\times\vec{ F }$',isolate=[r'\vec{ L }',r'\vec{ M }',r'\vec{ r }',r'\vec{ F }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        allg = TexText("Allgemeiner für einen ausgedehnten Körper\n" + r'mit der Massendichte $\rho$:')
        allg_tex1 = TexText(r'$\frac{\mathrm{d} \vec{ L }}{\mathrm{d} t} = \int \mathrm{d}^3r\ \vec{ r }\times\vec{f}_{LT}$',isolate=[r'\vec{ L }',r'\vec{ r }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        allg_tex2 = VGroup(
            TexText(r'$\vec{f}_{LT} = \rho \vec{ v }\times\vec{ B }(\vec{ r }+\vec{r}_S)$',isolate=[r'\vec{ v }',r'\vec{ B }',r'\vec{ S }',r'\vec{ r }']).set_color_by_tex_to_color_map(symCols,only_isolated=True),
            TexText(r'$= \frac{\rho}{r_S^3} \left[2 \vec{ v }\times\vec{ S } - \frac{6 (\vec{ S }\cdot\vec{r}_S)}{r_S^2}\vec{ v }\times (\vec{ r }+\vec{r}_S)\right]$',isolate=[r'\vec{ v }',r'\vec{ B }',r'\vec{ S }',r'\vec{ r }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)).arrange()
        allg_tex3 = TexText(r'$\frac{\mathrm{d} \vec{ L }}{\mathrm{d} t} = \vec{ L }\times\vec{ \Omega }$',isolate=[r'\vec{ L }',r'\vec{ \Omega }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        allg_tex4 = TexText(r'$\vec{ \Omega } = \frac{\vec{ B }(\vec{ r }_S)}{2}$',isolate=[r'\vec{ \Omega }',r'\vec{ B }']).set_color_by_tex_to_color_map(symCols,only_isolated=True)
        align_mobjs([(drehimp,),(drehimp_tex,),(magn,),(magn_tex,),(lforce,),(lforce_tex,),(torque,),(torque_tex,)],self.slide_title)
        align_mobjs([(allg,),(allg_tex1,),(allg_tex2,),(allg_tex3,),(allg_tex4,)],self.slide_title)
        # add arrow from allg_tex1 to allg_tex3, shift allg_tex2 to right of that arrow
        arr = Arrow(start=allg_tex1[6].get_edge_center(DOWN), end=allg_tex3[6].get_edge_center(UP), buff=0.1, color=FRONTCOL).fix_in_frame()
        allg_tex2.shift(RIGHT)

        probe_arr.remove_updater(rot_spin)
        self.canvas_objs.remove(sphere)
        new_axes = ThreeDAxes(x_range=(-0.5,0.5,0.5),y_range=(-0.5,0.5,0.5),z_range=(-0.5,0.5,0.5))
        new_axes.apply_depth_test(recurse=True)
        self.play(
            self.next_slide_number_animation(),
            self.offset_3d.animate.set_value(2.4+0j),
            self.frame.animate.reorient(-20,58,0,(0,0,0),2),
            probe.animate.shift((-2,0,0)),
            probe_arr.animate.shift((-2,0,0)),
            Write(drehimp),
            Write(drehimp_tex),
            ReplacementTransform(axes,new_axes),
            self.wipe([sphere, lanim_1, lanim_2, precession_circle, precession_arrow],[],return_animation=True))
        self.pause(auto_next=True)
        
        
        # B-Vektoren einzeichnen
        bvecs = Group(*[mt.Arrow3D(start=(x,y,-0.25),end=(x,y,0.25),color=symCols[r'\vec{ B }']) for x in np.linspace(-0.5,0.5,2) for y in np.linspace(-0.5,0.5,2)])
        self.play(ShowCreation(bvecs),Write(magn),Write(magn_tex))
        self.pause(loop=True)


        self.wait(np.pi/spin)
        self.pause(auto_next=True)


        # Kreisel 2 geschw. vektoren einzeichnen
        velvecposs = [np.array(rotate_vector((0,0,0.2),np.pi/2+precession_angle,axis=UP))]
        velvecposs = [(velvecposs[0],velvecposs[0]+np.array([0,0.3,0])),(-velvecposs[0],-velvecposs[0]-np.array([0,0.3,0]))]
        velvecs = Group(*[mt.Arrow3D(start=start,end=end,tip_length=0.06,color=symCols[r'\vec{ v }']) for start,end in velvecposs])
        self.play(ShowCreation(velvecs))
        self.pause(loop=True)


        self.wait(np.pi/spin)
        self.pause(auto_next=True)


        # resultierende Kraft einzeichnen
        forcevecposs = [(velvecposs[0][0],velvecposs[0][0]+np.array([0.3,0,0])),(velvecposs[1][0],velvecposs[1][0]-np.array([0.3,0,0]))]
        forcevecs = Group(*[mt.Arrow3D(start=start,end=end,tip_length=0.06,color=symCols[r'\vec{ F }']) for start,end in forcevecposs])
        self.play(ShowCreation(forcevecs),Write(lforce),Write(lforce_tex))
        self.pause(loop=True)


        self.wait(np.pi/spin)
        self.pause(auto_next=True)


        # resultierendes Drehmoment einzeichnen
        posposs = [(np.zeros(3),velvecposs[0][0]),(np.zeros(3),velvecposs[1][0])]
        posvecs = Group(*[mt.Arrow3D(start=start,end=end,tip_length=0.06,color=symCols[r'\vec{ r }']) for start,end in posposs])
        torquepos = np.array(rotate_vector((0,0,0.5),precession_angle,axis=UP))
        torquevecposs = [(torquepos,torquepos+np.array([0,0.2,0])),(torquepos+np.array([0,0.2,0]),torquepos+np.array([0,0.4,0]))]
        torquevecs = Group(*[mt.Arrow3D(start=start,end=end,tip_length=0.06,color=symCols[r'\vec{ M }']) for start,end in torquevecposs])
        self.play(Uncreate(probe))
        self.play(ShowCreation(posvecs),ShowCreation(torquevecs),Write(torque),Write(torque_tex))
        self.pause(loop=True)


        self.wait(np.pi/spin)
        self.pause(auto_next=True)


        precession_arrow = mt.Arrow3D(start=(0,0,0),end=(0,0,0.462),tip_width_ratio = 0.3,tip_length = 0.1,shaft_width = 0.015,color=symCols[r'\vec{ \Omega }'],depth_test=True)
        precession_circle = Circle(radius=0.191, color=symCols[r'\vec{ \Omega }'], fill_opacity=0.4, stroke_width=DEFAULT_STROKE_WIDTH).rotate(np.pi,axis=UP)
        precession_circle.move_to((0,0,0.462))
        self.play(ShowCreation(precession_arrow),ShowCreation(precession_circle))
        self.pause(loop=True)


        for vecs in [torquevecs,posvecs,velvecs,forcevecs,probe_arr]:
            vecs.add_updater(lambda obj,dt: obj.rotate(dt*np.pi/4,axis=(0,0,-1),about_point=(0,0,0)))
        self.wait(8.0)
        self.pause()


        #   Präzession allgemein (91-97)
        self.setup_new_slide(title='Präzession',cleanup=True)
        self.play(Write(allg))
        self.pause()


        self.play(Write(allg_tex1))
        self.pause()


        self.play(Write(arr),Write(allg_tex2[0]))
        self.pause()


        self.play(Write(allg_tex2[1]))
        self.pause()


        self.play(Write(allg_tex3))
        self.pause()


        self.play(Write(allg_tex4))
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