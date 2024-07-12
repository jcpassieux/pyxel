# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:45:49 2024

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px

f = px.Image('img.png').Load()
f.pix[:10, :] = 0 
f.pix[-15:, :] = 0
f.pix[:, -10:] = 0
f.pix[:, :10] = 0
f.GaussianFilter(0.8)

f.Plot()

m, cam = px.MeshFromImage(f, 128, 5, appls=0.5, typel='tri')

m.Plot()

px.PlotMeshImage(f, m, cam)
