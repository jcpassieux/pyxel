#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021    

    Example 2 : BASIC
    Analyse an image series.

    """

import os
import numpy as np
import pyxel as px

imnums = np.array([53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 75])
imagefile = os.path.join('data', 'dic_composite', 'zoom-0%03d_1.tif')
imref = imagefile % imnums[0]
f = px.Image(imref).Load()

m = px.ReadMeshINP(os.path.join('data', 'dic_composite', 'olfa3.inp'))

p=np.array([ 1.05449047e+04,  5.12335842e-02, -9.63541211e-02, -4.17489457e-03])
cam=px.Camera(p)

m.Connectivity()
m.DICIntegration(cam)

UU=px.CorrelateTimeIncr(m, f, imagefile, imnums, cam, [3, 2, 1, 0])

m.VTKSolSeries('example_2', UU)

