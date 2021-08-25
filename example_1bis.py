#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021   

    Example 1 : BASIC
    Analyse only one image.

        """

import os
import numpy as np
import pyxel as px

imref = os.path.join('data', 'dic_composite', 'zoom-0053_1.tif')
f = px.Image(imref).Load()
imdef = os.path.join('data', 'dic_composite', 'zoom-0070_1.tif')
g = px.Image(imdef).Load()

m = px.ReadMeshGMSH(os.path.join('data', 'dic_composite', 'gmsh_t3_mm.msh'))
m.Plot()

#%% CAMERA MODEL

do_calibration = False

if do_calibration:
    ls = px.LSCalibrator(f,m)
    ls.NewCircle()
    ls.NewLine()
    ls.NewLine()
    ls.FineTuning()
    cam=ls.Calibration()
else:
    # reuse previous calibration parameters
    cam = px.Camera(np.array([-10.513327, -50.987166, 6.449013, 4.709107]))

px.PlotMeshImage(f,m,cam)

#%% 

m.Connectivity()
m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res=px.Correlate(f ,g, m, cam, U0=U)


m.PlotContourDispl(U,s=30)
