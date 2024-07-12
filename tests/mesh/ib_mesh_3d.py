# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:45:49 2024

@author: passieux
"""

import numpy as np  
import pyxel as px

# building a virtual volume image of a cylinder
f = px.Volume('')
x = np.arange(80)
y = np.arange(80)
z = np.arange(100)
X, Y, Z = np.meshgrid(x, y, z)
f.pix = (((X-40)**2 + (Y-40)**2) < 35**2) * 255
f.pix[:, :, :5] = 0
f.pix[:, :, -5:] = 0
f.GaussianFilter(0.8)
f.Plot()

m, cam = px.MeshFromImage3D(f)

m.Plot()

px.PlotMeshImage3d(f, m, cam)
