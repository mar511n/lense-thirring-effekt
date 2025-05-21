#!/bin/sh

manim-slides render --GL --config_file config.yml hauptseminar_gl.py LenseThirringGL
manim-slides convert --to pdf LenseThirringGL lense-thirring.pdf
manim-slides convert --to html --one-file LenseThirringGL lense-thirring.html
cp lense-thirring.html ../index.html
manim-slides present --show-info-window LenseThirringGL