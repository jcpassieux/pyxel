#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 6 : ADVANCED
    Implement the different Exact GN solver.

    """

import numpy as np
import scipy.sparse.linalg as splalg
import pyxel as px


f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

m = px.ReadMesh('abaqus_q4_m.inp')

cam = px.Camera(2)
cam.set_p([3.144718, 0.096486, 0.081304, 0.000095])

m.Connectivity()
m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], l0=0.002)

dic = px.DICEngine()
for k in range(30):
    H = dic.ComputeLHS2(f, g, m, cam, U)
    b, res = dic.ComputeRHS2(g, m, cam, U)
    dU = splalg.spsolve(H, b)
    U += dU
    err = np.linalg.norm(dU) / np.linalg.norm(U)
    print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (k + 1,
                                                           np.std(res), err))
    if err < 1e-3:
        break

m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)
