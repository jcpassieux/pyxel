# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:14:37 2023

@author: passieux
"""

import numpy as np
import pyxel as px

f = px.Image('zoom-0053_1.tif').Load()
# f.Plot()
g = px.Image('zoom-0070_1.tif').Load()
# g.Plot()

# f.SelectROI()
roi = np.array([[536, 54], [849, 481]])
m, cam = px.SplineFromROI(roi, dx=30, degree=[2, 2])
# px.PlotMeshImage(f, m, cam)

m.Connectivity()
# m.Plot()

m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

m.Plot(alpha=0.5)
m.Plot(U=3*U)

m.VTKSol('toto_spline', U, n=[50, 50])


# %% Same with a non square spline domain

a = 0.925
ri = 0.5
re = 1.58
Xi = np.array([[ri, 0.5*(re-ri), re],
               [ri*a, 0.5*(re-ri)*a, re*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [ri*a, 0.5*(re-ri)*a, re*a],
               [ri, 0.5*(re-ri), re]])

ctrlPts = np.array([Xi, Yi])
degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

n = 5
newr = np.linspace(0, 1, n+2)[1:-1]
n = 10
newt = np.linspace(0, 1, n+2)[1:-1]
m = px.BSplinePatch(ctrlPts, degree, knotVect)
m.KnotInsertion([newt, newr])
# m.DegreeElevation(np.array([3, 3]))
m.Plot()

cam = px.Camera(2)
cam.set_p([0, 6.95, 5.36, 1/100])
px.PlotMeshImage(f, m, cam)

m.Connectivity()
m.DICIntegration(cam)
U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

m.Plot(U, alpha=0.5)
m.Plot(U=3*U)

px.PlotMeshImage(g, m, cam, U)
m.VTKSol('toto_spline', U, n=[50, 50])
