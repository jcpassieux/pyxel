#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2026

    Example Stereo: calibration of the camera

"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px


# %%

board = px.calibration.Board((12, 9), 'circles', 3e-3)
# board.Plot()

cam0 = px.calibration.Camera('calibration/cam0')
cam1 = px.calibration.Camera('calibration/cam1')
opt = dict()
opt['median'] = 15
opt['cb_orientation'] = False

cam0.DetectPoints(board, opt)
cam1.DetectPoints(board, opt)

cam0.Calibrate(board)
cam1.Calibrate(board)

rig = px.calibration.StereoRig(cam0, cam1)
rig.StereoCalibration(board)

# %% saving
cam0.SaveParams('calibration/cam0_params')
cam1.SaveParams('calibration/cam1_params')
rig.SaveParams('calibration/Calirbration_params')