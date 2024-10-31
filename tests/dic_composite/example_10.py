#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2021

    Example 10 : BASIC
    Initialize with DIS OPtical Flow

        """

import pyxel as px

f = px.Image('zoom-0053_1.tif').Load()
g = px.Image('zoom-0070_1.tif').Load()
m = px.ReadMesh('abaqus_q4_m.inp')
m.Connectivity()

cam = px.Camera(2)
cam.set_p([-1.573863, 0.081188, 0.096383, 0.000095])

U0 = px.DISFlowInit(f, g, m, cam)
U, res = px.Correlate(f, g, m, cam, U0=U0)

m.Plot(edgecolor='#CCCCCC')
m.Plot(U0, 30, edgecolor='red')
m.Plot(U, 30)
