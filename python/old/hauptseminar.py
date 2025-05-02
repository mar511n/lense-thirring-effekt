from manim import *
#import manim.opengl as mgl
#from manim_slides import ThreeDSlide
from manim_slides import Slide
#import scipy.interpolate as spint
import json
import os

"""
Strukturierung:

?   Einleitende Dinge zu ART (Geodäten & Metrik, Einsteingl. -> Metrik)
??  Linearisierung Einsteingl. (Annahme tau=t sagen)
    resultierende Gl. <-> e-dynamik (erst maxwell, dann Coulomb-Kraft (ersetzen der B,E Felder))
    Problem einer rotierenden Kugelmasse (dichte und stromdichte hinschreiben) (sagen, dass Lösung wie schon in E-dynamik is)
?   Darstellung der E,B Felder
    Alternative Darstellung über die Raumzeit
??  Kerr-Metrik
"""
Inhalt_strs = ('ART', 'Linearisierung', 'Gravitoelektromagnetismus', 'Rotierende Kugelmasse', 'EM-Felder', 'Raumzeitdarstellung', 'Kerr-Metrik')

def paragraph(*strs, alignment=LEFT, direction=DOWN, **kwargs):
    """
    Constructs a Group of tex mobjects that are aligned and ordered
    example: paragraph("hello world","hello world 2")
    """
    texts = VGroup(*[Tex(s, **kwargs) for s in strs]).arrange(direction)
    if len(strs) > 1:
        for text in texts[1:]:
            text.align_to(texts[0], direction=alignment)
    return texts

class MyCamera(ThreeDCamera):
    def transform_points_pre_display(self, mobject, points):
        if getattr(mobject, "fixed", False):
            return points
        else:
            return super().transform_points_pre_display(mobject, points)
      
class MyThreeDScene(ThreeDScene):
    def __init__(self, camera_class=MyCamera, ambient_camera_rotation=None,
                 default_angled_camera_orientation_kwargs=None, **kwargs):
        super().__init__(camera_class=camera_class, **kwargs)    

def make_fixed(*mobs):
    for mob in mobs:
        mob.fixed = True
        for submob in mob.family_members_with_points():
            submob.fixed = True

class LenseThirring(Slide, MyThreeDScene):
    def __init__(self, camera_class=MyCamera, ambient_camera_rotation=None,
                 default_angled_camera_orientation_kwargs=None, **kwargs):
        super().__init__(camera_class=camera_class, **kwargs)
    def construct(self):
        # Font sizes
        self.TITLE_FONT_SIZE = 48
        self.CONTENT_FONT_SIZE = 0.8 * self.TITLE_FONT_SIZE

        # make slide number & title & init LaTex
        self.slide_number = Integer(1, font_size=self.CONTENT_FONT_SIZE).to_corner(DR)
        self.slide_title = Tex(r"Lense-Thirring-Effekt", font_size=self.TITLE_FONT_SIZE).to_corner(UL)
        make_fixed(self.slide_number)
        make_fixed(self.slide_title)
        self.add_to_canvas(slide_number=self.slide_number, slide_title=self.slide_title)
        self.add(self.slide_number, self.slide_title)
        
        self.tex_template = TexTemplate()
        self.tex_template.add_to_preamble(
            r"""
        \usepackage{amsfonts}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage[T1]{fontenc}
        \usepackage{mathtools}
        \usepackage{bm}
        \usepackage[separate-uncertainty=true]{siunitx}
        \usepackage{upgreek}
        """
        )
        self.pause()
        
        Inhalt = BulletedList(*Inhalt_strs, buff=MED_SMALL_BUFF,font_size=self.CONTENT_FONT_SIZE)
        make_fixed(Inhalt)
        Inhalt.to_edge(LEFT)
        for istr in Inhalt_strs:
            self.play(Write(Inhalt.get_part_by_tex(istr),run_time=0.5))
            self.pause()
        
        # stuff zu ART und Linearisierung


        # Kraftformel für den lense-thirring Effekt
        #force_formula = MathTex(r'\frac{d \vec{v}}{dt}', r' = \frac{1}{r^3} \Bigl[', r'-M \vec{r}', r' + \vec{v}\times\vec{S}',r' - \frac{3 (\vec{S}\cdot\vec{r})}{r^2} \vec{v}\times\vec{r}', r'\Bigr]', font_size=self.CONTENT_FONT_SIZE)
        #make_fixed(force_formula)
        #self.play(Write(force_formula))

        # Darstellung E,B Feld
        self.setup_new_slide(title=r'EM-Felder',cleanup=True)
        axes = ThreeDAxes(
            x_range=(-4,4,1),
            y_range=(-4,4,1),
            z_range=(-4,4,1),
            x_length=8,
            y_length=8,
            z_length=8,
        )
        self.set_camera_orientation(phi=50 * DEGREES, theta=70 * DEGREES, zoom=1.3)
        self.play(Write(axes))

        self.pause()
        self.begin_ambient_camera_rotation(rate=-0.3)
        
        def efield(rv):
            r = np.linalg.norm(rv)
            if r < 1:
                return 0*rv
            return -rv/r**3
        
        def bfield(rv):
            r = np.linalg.norm(rv)
            if r < 1:
                return 0*rv
            S = np.array([0,0,1])
            return (S-3*np.dot(S,rv)/r**2 * rv)/r**3

        e_vec_field = StreamLines(
            efield,
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            stroke_width=2.5,
            padding=1,
            #length_func=lambda l: 0.4*(1-np.exp(-l)),
            #colors=[PURE_BLUE,PURE_RED],
            dt=1e-2,
        )
        self.add(e_vec_field)
        e_vec_field.start_animation(warm_up=False,flow_speed=1.5,time_width=1)
        self.wait(4.0)

        self.pause()
        
        b_vec_field = StreamLines(
            bfield,
            x_range=[-4, 4, 1/2],
            y_range=[-4, 4, 1/2],
            z_range=[-4, 4, 1/2],
            stroke_width=2.5,
            padding=1,
            #length_func=lambda l: 0.4*(1-np.exp(-l)),
            #colors=[PURE_BLUE,PURE_RED],
            dt=1e-2,
        )
        self.remove(e_vec_field)
        self.add(b_vec_field)
        b_vec_field.start_animation(warm_up=False,flow_speed=1.5,time_width=1)
        #self.play(e_vec_field.animate.become(b_vec_field))
        self.wait(4.0)
        
        self.pause()

    def pause(self):
        self.wait(0.1)
        self.next_slide()

    def next_slide_number_animation(self):
        return self.slide_number.animate(run_time=0.5).set_value(
            self.slide_number.get_value() + 1
        )

    def next_slide_title_animation(self, title):
        newT = Tex(title, font_size=self.TITLE_FONT_SIZE).move_to(self.slide_title).align_to(self.slide_title, LEFT)
        make_fixed(newT)
        return Transform(
            self.slide_title,
            newT,
        )

    def setup_new_slide(self, title, cleanup=False, contents=None):
        if cleanup:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
                self.wipe(
                    self.mobjects_without_canvas,
                    contents if contents else [],
                    return_animation=True,
                ),
            )
        else:
            self.play(
                self.next_slide_number_animation(),
                self.next_slide_title_animation(title),
            )
        make_fixed(self.slide_number)


"""
def load_setup(num, time_tracker, axes, color=BLACK):
    all_setups = {}
    if os.path.isfile('all_setups.json'):
        try:
            with open('all_setups.json', 'r') as fp:
                all_setups = json.load(fp)
        except:
            print("could not open all_setups.json")
    setup = all_setups.get(f"setup {num}")
    if setup == None:
        return None
    lines = np.load(f'./spacetime_sims/spacetime_lines{num}.npy')
    times = np.load(f'./spacetime_sims/lines{num}_timestamps.npy')
    print(f'loaded lines {lines.shape} and times {times.shape}')
    pfs = VGroup()
    for line in lines:
        pfs.add(TimeAnimatedInterpolatedCurve(time_tracker, line, times, axes, color))
    return (setup, pfs)

def TimeAnimatedInterpolatedCurve(time_tracker, lines, times, axes, color=BLACK):
    intf = spint.interp1d(times, lines.T)
    pf = ThreeDVMobject(shade_in_3d=True).set_points_as_corners(axes.c2p(intf(time_tracker.get_value()).T))
    pf.set_color(color)
    pf.add_updater(lambda m: m.set_points_as_corners(axes.c2p(intf(time_tracker.get_value()).T)))
    return pf
"""