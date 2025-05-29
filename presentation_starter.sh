#!/bin/bash
export XCURSOR_SIZE=80
cd python
manim-slides present LenseThirringGL & ../tcpresenter
unset XCURSOR_SIZE
