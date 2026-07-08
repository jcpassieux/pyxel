# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:43:16 2022

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px

# %% Loading the FE mesh

m = px.ReadMesh('heatsink.inp')
# m.Plot()
m.Connectivity()
m.GaussIntegration()

T = np.zeros(len(m.n))

# Heat parameters
k = 237.  # W/(mK)
h = 70  # W/(m2K)
t_c = 100
t_air = 20

K = m.GetConductionOperator(k)
H = m.GetConvectionOperator(h)

# FE simulation
rep = m.SelectEndLine('left', 1e-3)
BC = [[rep, [[0, t_c], ]]]

Kd, Fd, Ud = m.ApplyDirichlet(K, BC, 'penalty')

Tair = np.ones(len(m.n)) * t_air
Ktot = Kd + H
btot = Fd + H @ Tair

T = m.LinearSolver(Ktot, btot)
m.PlotContourDispl(T, s=0, stype='mag', cmap='coolwarm')


# %%

E = 70e9
nu = 0.27
C = px.Hooke([E, nu])
alpha_th = 23e-6  # unit °C^(-1)

K = m.Stiffness(C)

# Computing thermal strain under plane stress assumption
Eth = alpha_th * E/(1-nu) * (T - t_air)
Ethn = m.DOF2GP(np.hstack((Eth, Eth)))
Eths = Ethn * 0
Fth = m.ComputeInternalForce(Ethn, Eths)

BC = [[rep, [[0, 0], [1, 0]]]]
Kd, Fd, Ud = m.ApplyDirichlet(K, BC, 'penalty')

# with clampling
U = m.LinearSolver(Kd, Fth, Fd)

# withour clamping
U = m.LinearSolver(K, Fth)

m.PlotContourDispl(U, s=30, stype='comp', cmap='rainbow')

plt.figure()
m.Plot(alpha=0.3)
m.Plot(U, 300)

# %%
En_total, Es_total = m.StrainAtNodes(U)
m.PlotContourTensorField(U, En_total, Es_total, stype='comp', field_name='Total Strain \epsilon')
m.PlotContourTensorField(U, Ethn, Eths, stype='comp', field_name='Thermal Strain \epsilon')

Ex = En_total[:, 0] - alpha_th * (T - t_air)
Ey = En_total[:, 1] - alpha_th * (T - t_air)
Exy = Es_total[:, 0]
Eeln = np.c_[Ex, Ey]
Eels = np.c_[Exy, 0*Exy]
m.PlotContourTensorField(U, Eeln, Eels, stype='comp', field_name='Elastic_ Strain')

Sign, Sigs = px.Strain2Stress(C, Eeln, Eels)
m.PlotContourTensorField(U, Sign, Sigs, stype='comp', field_name='SIG')
