# 🌀 Lense-Thirring Effect Visualization with ManimGL

[![License](https://img.shields.io/github/license/yourusername/yourrepo)](https://github.com/yourusername/yourrepo/blob/main/LICENSE)
[![Manim Version](https://img.shields.io/badge/Manim-GL-blue)](https://github.com/3b1b/manim)

A visual exploration of the **Lense-Thirring effect** (frame-dragging in general relativity) using [ManimGL](https://github.com/3b1b/manim). This repository contains code for creating animated explanations of relativistic spacetime effects, along with supporting simulations.

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
- `manim-slides`, `numpy`, `scipy`, `matplotlib`


### Rendering Animations
```bash
# Example: Render the frame-dragging animation
manim -pql lense_thirring_scenes.py FrameDraggingScene

# Available scenes:
# - KerrSpacetimeScene
# - GyroscopePrecessionScene
# - GravitomagnetismScene
```

---

## 📁 Project Structure
```
yourrepo/
├── animations/          # ManimGL scene scripts
├── simulations/         # Python calculation modules
├── assets/              # SVG/MathTex assets
├── renders/             # Output video directory (gitignored)
├── .gitignore           # Standard Python + Manim ignores
└── README.md
```

---

## 📸 Preview

![Frame Dragging Animation](assets/preview.gif)
*Visualization of spacetime frame-dragging near a rotating black hole*

---

## 📚 License
This project is licensed under the [MIT License](LICENSE), allowing free use for educational and commercial purposes.
