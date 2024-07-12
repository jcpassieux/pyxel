#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 2 : BASIC
    Analyse an image series.

    """

import numpy as np
import pyxel as px

imnums = np.array([53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 75])
imagefile = 'zoom-0%03d_1.tif'
imref = imagefile % imnums[0]
f = px.Image(imref).Load()

m = px.ReadMesh('abaqus_q4_m.inp')

cam = px.Camera(2)
cam.set_p([-1.573863, 0.081188, 0.096383, 0.000095])

m.Connectivity()
m.DICIntegration(cam)

UU = px.CorrelateTimeIncr(m, f, imagefile, imnums, cam, [3, 2, 1, 0])

m.AnimatedPlot(UU, 30)

m.VTKSolSeries('example_2', UU)
