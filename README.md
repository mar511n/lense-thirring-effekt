# 🌀 Lense-Thirring Effect Visualization with ManimGL

[![License](https://img.shields.io/github/license/mar511n/lense-thirring-effekt)](https://github.com/mar511n/lense-thirring-effekt/blob/main/LICENSE)
[![Manim Version](https://img.shields.io/badge/Manim-GL-blue)](https://github.com/3b1b/manim)
[![Manim Version](https://img.shields.io/badge/Manim-Slides-blue)](https://github.com/jeertmans/manim-slides)

A visual exploration of the **Lense-Thirring effect** (frame-dragging in general relativity) using [ManimGL](https://github.com/3b1b/manim) and [Manim-Slides](https://github.com/jeertmans/manim-slides). This repository contains code for a manim-slides presentation with animated explanations of relativistic spacetime effects, along with supporting simulations.

---

## 🎥 What's Inside

- **ManimGL Animations**: Visualizations of:
  - Gravitomagnetic fields
  - Geodesics in curved manifolds
  - Rotating black hole spacetime geometry
  - Gyroscope precession near rotating masses
- **Python Simulations**: Numerical calculations for:
  - General geodesics on 2D manifolds
  - Frame-dragging forces
- **Educational Resources**: Diagrams and derivations for teaching

---

## 🛠️ Setup & Usage

### Prerequisites
- Python 3.8+
- [ManimGL](https://github.com/3b1b/manim) (`pip install manimlib`)
- [Manim-Slides](https://github.com/jeertmans/manim-slides) (`pip install "manim-slides[manimgl]"`)
- `numpy`, `scipy`, `matplotlib`


### Rendering Animations
```bash
# Example: Render the whole presentation
./build_show_presentation.sh

# Available scene:
# - LenseThirringGL
# use the start_at_animation_number and end_at_animation_number to render subsections
```

---

## 📁 Project Structure
```
lense-thirring-effekt/
├── renders/			# complete renders of the presentation
├── Mathematica/        # Mathematica code for some simulations & calculations
├── python/             # all files for manim and the presentation (hauptseminar_gl.py)
├── Abstract/           # Latex code & PDF for an abstract
├── papers/           	# interesting papers
├── index.html			# the presentation in html format
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📸 Preview

![Frame Dragging Animation](renders/preview.mp4)
*Visualization of geodesics and metric in curved space*

---

## 📚 License
This project is licensed under the [MIT License](LICENSE), allowing free use for educational and commercial purposes.
