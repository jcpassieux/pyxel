#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 1 : BASIC
    Analyse only one image with a INP Qua4 mesh in m.

"""

import numpy as np
import pyxel as px

# %% IMAGES
f = px.Image('zoom-0053_1.tif').Load()
f.Plot()
g = px.Image('zoom-0070_1.tif').Load()
g.Plot()

# %% MESH
m = px.ReadMesh('abaqus_q4_m.inp')
m.Plot()

# %% CAMERA MODEL

do_calibration = False

if do_calibration:
    ls = px.LSCalibrator(f, m)
    ls.NewCircle()
    ls.NewLine()
    ls.NewLine()
    ls.FineTuning()
    ls.Init3Pts()
    cam = ls.Calibration()
else:
    # reuse previous calibration parameters
    cam = px.Camera(2)
    cam.set_p([-1.573863, 0.081188, 0.096383, 0.000095])

px.PlotMeshImage(f, m, cam)

# %% Pre-processing
m.Connectivity()
m.DICIntegration(cam)

# %% Do Correlation
U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res = px.Correlate(f, g, m, cam, U0=U)

# %%  Post-processing
# Visualization: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U, 30)

# Visualization: displacement fields
m.PlotContourDispl(U, s=30)

# Visualization: strain fields
m.PlotContourStrain(U, clim=1, cmap='RdBu')

# Plot deformed Mesh on deformed state image
px.PlotMeshImage(g, m, cam, U)

# Plot residual
m.PlotResidualMap(res, cam, 1e5)

# Export for Paraview
m.VTKSol('example_1', U)

# Export for Paraview at integration points (residual map)
m.VTKIntegrationPoints(cam, f, g, U)

# %% Post-processing as a pixel map

# Initialization
emp = px.ExportPixMap(f, m, cam)

# Get Residual on the pixel map
Rmap = emp.PlotResidual(f, g, U)

# Get displacement on the pixel map
Umap, Vmap = emp.PlotDispl(U)
