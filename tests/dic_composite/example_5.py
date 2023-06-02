#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021  

    Example 5:
        Other element types.

     """

import numpy as np
import pyxel as px
import matplotlib.pyplot as plt

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()

# f.SelectROI()
roi = np.array([[ 537,   24], [ 850,  488]])
m = dict()
m[0], cam = px.MeshFromROI(roi, 50, f, typel=2)# tri3
m[1], _ = px.MeshFromROI(roi, 50, f, typel=3)  # qua4
m[2], _ = px.MeshFromROI(roi, 50, f, typel=9)  # tri6
m[3], _ = px.MeshFromROI(roi, 50, f, typel=10) # qua9
m[4], _ = px.MeshFromROI(roi, 50, f, typel=16) # qua8

for k in m.keys():
    px.PlotMeshImage(f, m[k], cam)
    m[k].Connectivity()
    m[k].DICIntegration(cam)
    U = px.MultiscaleInit(f, g, m[k], cam, scales=[3, 2, 1], l0=30)
    U, res = px.Correlate(f, g, m[k], cam, U0=U)
    plt.figure()
    m[k].Plot(edgecolor='#CCCCCC')
    m[k].Plot(U,30)
