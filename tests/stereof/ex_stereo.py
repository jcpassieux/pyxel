#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2026

    Example Stereo Optical Flow (no Finite Elements here !)

"""


import pyxel as px


# %% Loading Calibration

cam0 = px.calibration.Camera('')
cam0.LoadParams('calibration/cam0_params.npz')
cam1 = px.calibration.Camera('')
cam1.LoadParams('calibration/cam1_params.npz')
rig = px.calibration.StereoRig(cam0, cam1)
rig.LoadParams('calibration/Calirbration_params.npz')


# %%
leftimagefiles = 'openhole-%04d_0.tif'
rightimagefiles = 'openhole-%04d_1.tif'
imfiles = [leftimagefiles, rightimagefiles]
mask = px.Image('mask.png').Load()
mask.Plot()

SDIC = px.StereOF(rig, imfiles, mask)

X = SDIC.ShapeMeasurement()
U = SDIC.DisplacementMeasurement(2)

# %% Postpro
im_c0_t0, im_c1_t0 = SDIC.GetImgAtPoints(SDIC.X_t0, 0)
fields = {'shape': X,
          'image_c0_t0': im_c0_t0,
          'residual_c01': im_c1_t0-im_c0_t0, 
          }
SDIC.VTKfile('shape', fields)

im_c0_t2, im_c1_t2 = SDIC.GetImgAtPoints(SDIC.X_t0 + U, 2)
fields = {'displ1': U,
          'residual_c0': im_c0_t2-im_c0_t0,
          'residual_c1': im_c1_t2-im_c1_t0,
          }
SDIC.VTKfile('displacements', fields)
