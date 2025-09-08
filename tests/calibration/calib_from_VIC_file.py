# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:52:22 2024

@author: passieux
"""

import pyxel.calibration as clb
import numpy as np

filename = 'calibration_sys1.txt'

rig = clb.CamerasFromVICFile(filename)
cam0 = rig.cam0
cam1 = rig.cam1

cam0.PlotParamsVIC(0)
cam1.PlotParamsVIC(1)
rig.PlotParamsVIC()

# %% test
pts1 = np.array([[1606., 1496.]])
pts2 = np.array([[1611., 1233.]])

# triangulation
X3d = rig.Triangulation(pts1, pts2)

# projection
uv0, uv1 = rig.Projection(X3d)

