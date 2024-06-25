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

cam = px.Camera(2)
cam.set_p([3.144718, 0.096486, 0.081304, 0.000095])

m.Connectivity()
m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])

l0 = 0.005
L = m.Laplacian()
U, res = px.Correlate(f, g, m, cam, U0=U, L=L, l0=l0)

m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)
