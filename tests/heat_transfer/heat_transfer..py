# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:43:16 2022

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px
import scipy as sp
import scipy.sparse.linalg as spl


def GetConductionOperator(m, k):
    K = k * m.Laplacian()
    rep = np.arange(m.ndof // 2)
    return K[np.ix_(rep, rep)]


def GetConvectionOperator(m, h):
    mb = m.BuildBoundaryMesh()
    nzv = 4 * len(mb.e[1])
    col = np.zeros(nzv)
    row = np.zeros(nzv)
    val = np.zeros(nzv)
    nzv = 0
    for i in range(len(mb.e[1])):
        n1 = mb.e[1][i][0]
        n2 = mb.e[1][i][1]
        L = np.linalg.norm(np.diff(m.n[mb.e[1][i]], axis=0))
        row[nzv + np.arange(4)] = np.array([n1, n1, n2, n2])
        col[nzv + np.arange(4)] = np.array([n1, n2, n1, n2])
        val[nzv + np.arange(4)] = np.array([2., 1., 1., 2.]) * L/6
        nzv += 4
    shape = (m.ndof // 2, m.ndof // 2)
    H = h * sp.sparse.csc_matrix((val, (row, col)), shape=shape)
    return H

def GetHeatEquivalentForce(m, DT):
    DTpg = m.phix[:, :m.ndof//2] @ DT
    Eth = alpha_th * DTpg
    bth = E/(1-nu) * m.dphixdx.T @ (m.wdetJ * Eth) + \
          E/(1-nu) * m.dphiydy.T @ (m.wdetJ * Eth)
    return bth
 
    
# %% 
"""
SOLVING THE HEAT TRANSFER PROBLEM
"""

m = px.ReadMesh('heatsink.inp')
# m.Plot()
m.Connectivity()
m.GaussIntegration()

T = np.zeros(len(m.n))

# Heat parameters
k = 237. # W/(mK)
h = 70  # W/(m2K)
t_c = 100
t_air = 20

# FE simulation with convection
K = GetConductionOperator(m, k)
H = GetConvectionOperator(m, h)

m.conn = m.conn[:, [0]]
m.ndof = m.ndof // 2
# FE simulation

rep = m.SelectEndLine('left', 1e-3)
BC = [ [rep, [[0, t_c] ]], ]

Kd, Fd, Ud = m.ApplyDirichlet(K, BC)


Tair = np.ones(m.ndof) * t_air
Ktot = Kd + H
btot = Fd + H @ Tair

T = m.LinearSolver(Ktot, btot)

m.PlotContourDispl(T, s=0, stype='mag', cmap='coolwarm')


# %%
"""
SOLVING THE MECHANICAL PB WITH THERMAL EXPANSION
"""

E = 70e9
nu = 0.27
C = px.Hooke([E, nu])
alpha_th = 23e-6  # unit °C^(-1)

m.Connectivity()
K = m.Stiffness(C)

T0 = np.ones(len(m.n)) * t_air
bth = GetHeatEquivalentForce(m, T-T0)

# with clampling
BC = [ [rep, [[0, 0], [1, 0] ]], ]
Kd, Fd, Ud = m.ApplyDirichlet(K, BC)

U = m.LinearSolver(Kd, bth)

# without clamping
U = spl.spsolve(K, bth)

m.PlotContourDispl(U, s=30, stype='mag', cmap='rainbow')

m.PlotContourStrain(U, s=30, stype='comp', clim=1.1)

plt.figure()
m.Plot(alpha=0.3)
m.Plot(U, 300)

# %%
"""
Computing STRESSES
"""

En_total, Es_total = m.StrainAtNodes(U)
m.PlotContourTensorField(U, En_total, Es_total, stype='comp', field_name='\epsilon')

Ex = En_total[:, 0] - alpha_th * (T-T0)
Ey = En_total[:, 1] - alpha_th * (T-T0)
Exy = Es_total[:, 0]

Sign, Sigs = px.Strain2Stress(C, np.c_[Ex, Ey], np.c_[Exy, 0*Exy])
m.PlotContourTensorField(U, Sign, Sigs, stype='comp', field_name='SIG')
