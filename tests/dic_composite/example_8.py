#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021

    Example 8 : ADVANCED
    Implement the inverse compositional Gauss Newton

    [REF] JC. Passieux and R. Bouclier. Classic and Inverse Compositional 
    Gauss-Newton in Global DIC. International Journal for Numerical Methods
    in Engineering, 119(6), p.453-468, 2019.

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

m.DICIntegration(cam, G=True)

U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], l0=0.003)

#%% ICGN

dic = px.DICEngine()
U = U0.copy()
H = dic.ComputeLHS(f, m, cam)
H_LU = splalg.splu(H)
repx = np.arange(m.ndof//2)
repy = np.arange(m.ndof//2, m.ndof)
phi = m.phix[:, repx]
MMLU = splalg.splu(phi.T @ phi)
dNdx = MMLU.solve((phi.T @ m.dphixdx[:, repx]).A)
dNdy = MMLU.solve((phi.T @ m.dphixdy[:, repx]).A)
for k in range(0,60):
    b, res = dic.ComputeRHS(g, m, cam, U)
    dU = H_LU.solve(b)
    # ICGN Correction:
    Ux = dNdx @ U[repx] * dU[repx] + dNdy @ U[repx] * dU[repy]
    Uy = dNdx @ U[repy] * dU[repx] + dNdy @ U[repy] * dU[repy]
    U += dU + np.append(Ux, Uy)
    err = np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc/dyn=%2.2f gl | dU/U=%1.2e" % (k + 1, np.std(res), err))
    if err<1e-3:
        break

m.Plot(U, alpha=0.2)
m.Plot(U, 30)
