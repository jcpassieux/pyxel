#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 4 :
    Use additional tikhonov regularization.

     """

import numpy as np
import pyxel as px

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

m = px.ReadMesh('abaqus_q4_m.inp')

p = np.array([1.05449047e+04, 5.12335842e-02,
             -9.63541211e-02, -4.17489457e-03])
cam = px.Camera(p)

m.Connectivity()
m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])

l0 = 0.005
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U, L=L, l0=l0)

m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)
