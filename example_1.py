#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021   

    Example 1 : BASIC
    Analyse only one image.

        """

import os
import numpy as np
import pyxel as px

#%% IMAGES

imref = os.path.join('data', 'dic_composite', 'zoom-0053_1.tif')
f = px.Image(imref).Load()
f.Plot()

imdef = os.path.join('data', 'dic_composite', 'zoom-0070_1.tif')
g = px.Image(imdef).Load()
g.Plot()

#%% MESH

m = px.ReadMeshINP(os.path.join('data', 'dic_composite', 'olfa3.inp'))
m.Plot()

#%% CAMERA MODEL

do_calibration = False

if do_calibration:
    ls = px.LSCalibrator(f,m)
    ls.NewCircle()
    ls.NewLine()
    ls.NewLine()
    ls.FineTuning()
    cam=ls.Calibration()
else:
    # reuse previous calibration parameters
    p = np.array([ 1.05449047e+04,  5.12335842e-02, -9.63541211e-02, -4.17489457e-03])
    cam = px.Camera(p)

px.PlotMeshImage(f,m,cam)

#%% Pre-processing 

m.Connectivity()
m.DICIntegration(cam)

#%% Do Correlation
U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1])
U, res=px.Correlate(f ,g, m, cam, U0=U)

#%%  Post-processing 

# Visualization: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U,30)

# Visualization: displacement fields
m.PlotContourDispl(U,s=30)

# Visualization: strain fields
m.PlotContourStrain(U)

# Plot deformed Mesh on deformed state image
px.PlotMeshImage(g,m,cam,U)

# Plot residual
m.PlotResidualMap(res,cam,1e5)

# Export for Paraview
m.VTKSol('example_1',U)
