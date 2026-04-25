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

cyl = (((X-40)**2 + (Y-40)**2) < 35**2) * (Z < 60)
sphere = ((X-40)**2 + (Y-40)**2 + (Z-60)**2) < 35**2
f.pix = (cyl+sphere) * 255
f.pix[:, :, :5] = 0
f.pix[:, :, -5:] = 0
f.GaussianFilter(0.8)
f.Plot()

f.Otsu(256)

# closed surface mesh
ms = f.MarchingCubes()

# building volume mesh
# m = px.MeshFromSurfaceCGAL(ms, facet_size=8, cell_size=8)
m = px.MeshFromSurfaceGMSH(ms, facet_size=5, cell_size=10)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])

m.KeepVolElems()
m.Plot()

m.Write('tmp.vtu')
px.PlotMeshImage3d(f, m, cam)




