#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2026

    Example 5(b) : RBM Initialize
    Use Fourier Rigid Body Translation for initialisation

    """

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px

# synthetise a pure rigid body translation from reference image
f = px.Image('zoom-0053_1.tif').Load()
g = f.Copy()
g.pix = g.pix[12:612, 450:950]
f.pix = f.pix[:600, 500:1000]

f.Plot()
plt.figure()
g.Plot()

m = px.ReadMesh('abaqus_q4_m.inp')
cam = px.Camera(2)
cam.set_p([-1.573863, 0.081188-0.0475, 0.096383, 0.000095])

roi = np.array([[36,   31], [351,  468]])
m.RemoveElemsOutsideRoi(roi, cam)
px.PlotMeshImage(f, m, cam)

m.Connectivity()
m.DICIntegration(cam)

pixel_shift = px.FourierRBT(f, g)

U0 = px.FourierRBT(f, g, m, cam)
U, res = px.Correlate(f, g, m, cam, U0=U0)

m.Plot(alpha=0.2)
m.Plot(U, 1)