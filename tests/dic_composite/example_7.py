#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 7 : ADVANCED
    Implement the different with element brightness and contrast correction.

    """

import numpy as np
import scipy.sparse.linalg as splalg
import pyxel as px

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

m = px.ReadMesh('abaqus_q4_m.inp')

p = np.array([1.05449047e+04, 5.12335842e-02,
              -9.63541211e-02, -4.17489457e-03])
cam = px.Camera(p)

m.Connectivity()
m.DICIntegration(cam, EB=True)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], l0=0.002)

dic = px.DICEngine()
H = dic.ComputeLHS_EB(f, m, cam)
H_LU = splalg.splu(H)
for k in range(30):
    b, res = dic.ComputeRHS_EB(g, m, cam, U)
    dU = H_LU.solve(b)
    U += dU
    err = np.linalg.norm(dU) / np.linalg.norm(U)
    print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (k + 1,
                                                           np.std(res), err))
    if err < 1e-3:
        break

m.PlotResidualMap(res, cam, 1e5)

U, res = px.Correlate(f, g, m, cam, U0=U)
m.PlotResidualMap(res, cam, 1e5)
