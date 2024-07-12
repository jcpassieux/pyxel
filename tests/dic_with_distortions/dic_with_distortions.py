# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:15:13 2024

@author: passieux
"""

import numpy as np
import pyxel as px

imagefiles = 'openhole-%04d_0.tif'
f = px.Image(imagefiles % 0).Load()
g = px.Image(imagefiles % 1).Load()

m = px.ReadMesh('abaqus_mesh_Q4.inp')
m.Connectivity()
m.Plot()

cam = px.Camera(dim=2)

# pre calibrated intrinsic params:
fx = 16372.27086046
fy = 16384.56096313
cx = 1792.01771429
cy = 773.42054646
k1 = 2.71725658e-02
k2 = 8.21579404e+00
k3 = -3.80725303e+02
p1 = -6.10518731e-03
p2 = 1.44932395e-02
cam.set_p(np.array([fx, fy, cx, cy]), 'intrinsic')
cam.set_p(np.array([k1, k2, p1, p2, k3]), 'distortion')

# Calibration of the extrinsics
calib_ext = False
if calib_ext:
    ls = px.LSCalibrator(f, m, cam)
    ls.NewCircle()
    ls.NewLine()
    ls.NewLine()
    ls.FineTuning()
    cam = ls.Calibration()
else:
    cam.set_p([0.011394, -0.018419, 0.007162, 0.493576])

px.PlotMeshImage(f, m, cam)

m.DICIntegration(cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

# Visualization: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)

# %%
# with distortions
n = np.array(cam.P(m.n[:, 0], m.n[:, 1])).T
m.Plot(n=n)

# without distortions
cam.set_p(np.array([0., 0, 0, 0, 0]), 'distortion')
n = np.array(cam.P(m.n[:, 0], m.n[:, 1])).T
m.Plot(n=n, edgecolor='r')
