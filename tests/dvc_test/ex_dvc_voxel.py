# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:21:25 2026

@author: passieux
"""
# %% 

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px



# %%
refname = 'dataset_0_binning2.npy'
defname = 'dataset_1_binning2.npy'

f = px.Volume(refname).Load()
g = px.Volume(defname).Load()
f.BuildInterp()
g.BuildInterp()

m = px.ReadMesh('conform_mesh_coarse.vtk', 3)
m = px.ReadMesh('conform_mesh_8.vtk', 3)
m.KeepVolElems()
m.RemoveUnusedNodes()
cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
px.PlotMeshImage3d(f, m, cam)

m.Connectivity()
U0 = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], l0=30, direct=False)

m.Plot(U0, 10)

# %% RUN SCALE 0
m.DVCIntegrationTetVoxel(f, cam)

L = m.Laplacian()
dic = px.DVCEngine()
H = dic.ComputeLHS(f, m, cam)
U, res = px.Correlate(f, g, m, cam, U0=U0, l0=12, L=L, H=H, dic=dic, direct=False)

# plot displacement in matplotlib
m.Plot(U, 10)

# plot deformed mesh in the deformed state image slices
px.PlotMeshImage3d(g, m, cam, U=U)

# plot the displacement and strain field for Paraview (in the Mesh CSYS and units)
m.VTKSol('displ_dvc', U)

# %% Export Residual map as a voxel image
evm = px.ExportVoxMap(f, m, cam)
R = evm.PlotResidual(f, g, U)
R_init = evm.PlotResidual(f, g, U*0)

h = px.Volume('')
h.pix = R_init
# plot slices in matplotlib
h.Plot(cmap='RdBu', vmin=-50, vmax=50)
plt.savefig('initial_residual.png')

h.pix = R
# plot slices in matplotlib
h.Plot(cmap='RdBu', vmin=-50, vmax=50)
plt.savefig('converged_residual.png')

# save as a volume tiff file
h.Save('residual.tiff')

# save as 3 VTI Slices
h.VTKSlice(dtype='float')


