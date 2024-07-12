#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 1bis : BASIC
    Analyse only one image with a GMSH Tri3 mesh in mm.

        """

import numpy as np
import pyxel as px

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

m = px.ReadMesh('gmsh_t3_mm.msh')
m.KeepSurfElems()  # remove elements of type segments and vertices
m.Plot()

# %% CAMERA MODEL

do_calibration = False

if do_calibration:
    ls = px.LSCalibrator(f, m)
    ls.NewCircle()
    ls.NewLine()
    ls.NewLine()
    ls.FineTuning()
    ls.Init3Pts()
    cam = ls.Calibration()
else:
    # reuse previous calibration parameters
    cam = px.Camera(2)
    cam.set_p([1.567737, 51.065538, 6.512059, 0.095166])

px.PlotMeshImage(f, m, cam)

# %%

m.Connectivity()
m.DICIntegration(cam)
m.DICIntegrationFast(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

m.PlotContourDispl(U, s=30)
