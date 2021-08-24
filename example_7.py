#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021

    Example 7 : ADVANCED
    Implement the different with element brightness and contrast correction.

    """

import os
import numpy as np
import scipy.sparse.linalg as splalg
import pyxel as px


imnums = np.array([53, 54, 57, 58, 61, 62, 65, 66, 69, 70, 75])
imagefile = os.path.join('data', 'dic_composite', 'zoom-0%03d_1.tif')

imref = imagefile % imnums[0]
f = px.Image(imref).Load()
imdef = imagefile % imnums[-2]
g = px.Image(imdef).Load()

m = px.ReadMeshINP(os.path.join('data', 'dic_composite', 'olfa3.inp'))

p=np.array([ 1.05449047e+04,  5.12335842e-02, -9.63541211e-02, -4.17489457e-03])
cam=px.Camera(p)

m.Connectivity()
m.DICIntegration(cam, EB=True)

U = px.MultiscaleInit(f, g, m, cam, scales=[3,2,1], l0=0.002)

dic = px.DICEngine()
H = dic.ComputeLHS_EB(f, m, cam)
H_LU = splalg.splu(H)
for k in range(30):
    b, res = dic.ComputeRHS_EB(g, m, cam, U)
    dU = H_LU.solve(b)
    U += dU
    err = np.linalg.norm(dU) / np.linalg.norm(U)
    print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (k + 1, np.std(res), err))
    if err < 1e-3:
        break

m.PlotResidualMap(res,cam,1e5)

U, res=px.Correlate(f ,g, m, cam, U0=U)
m.PlotResidualMap(res,cam,1e5)

