# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:43:16 2022

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px





# %% Loading the FE mesh
box = np.array([[0, 0, 0], [1, 1, 1]])
m = px.StructuredMeshHex8(box, 0.1)
# m.Plot()
m.Connectivity()
m.GaussIntegration()

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

T3d = np.hstack((T, T, T))
m.VTKSol('heat', T3d)


# %%

E = 70e9
nu = 0.27
C = px.Hooke([E, nu], 'isotropic_3D')
alpha_th = 23e-6  # unit °C^(-1)

K = m.Stiffness(C)

# Computing thermal strain under plane stress assumption
Eth = alpha_th * E/(1-nu) * (T - t_air)
Ethn = m.DOF2GP(np.hstack((Eth, Eth, Eth)))
Eths = Ethn * 0
Fth = m.ComputeInternalForce(Ethn, Eths)

BC = [[rep, [[0, 0], [1, 0], [2, 0]]]]
Kd, Fd, Ud = m.ApplyDirichlet(K, BC, 'penalty')

# with clampling
U = m.LinearSolver(Kd, Fth, Fd)

m.VTKSol('thermomechanical', U)


# %%
En_total, Es_total = m.StrainAtNodes(U)

Ex = En_total[:, 0] - alpha_th * (T - t_air)
Ey = En_total[:, 1] - alpha_th * (T - t_air)
Ez = En_total[:, 2] - alpha_th * (T - t_air)
Eeln = np.c_[Ex, Ey, Ez]
Eels = Es_total.copy()

Sign, Sigs = px.Strain2Stress(C, Eeln, Eels)

