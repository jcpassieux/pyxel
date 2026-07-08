# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:18:30 2026

Example 2: a plate with soft circular inclusions.
From an *.inp mesh already including element sets

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px


fn = 'rve_mesh.inp'
m = px.ReadMesh(fn)
m.KeepSurfElems()
m.Plot()

m.Write('test.vtu')
m.Connectivity()
m.GaussIntegration()

C = dict()
C['Set-1'] = px.Hooke([1e4, 0.3], 'isotropic_2D_ps')
C['Set-2'] = px.Hooke([1e1, 0.3], 'isotropic_2D_ps')
hooke = m.AssignMaterial2GaussPoint(C)
# hooke is now a (3 x 3 x npg) array instead of (3 x 3)

K = m.Stiffness(hooke)

# Dirichlet BC at y = 0
repb = m.SelectEndLine('bottom', 1e-5)
rept = m.SelectEndLine('top', 1e-5)

BC = [[repb, [[0, 0], [1, 0]]], ]   # setting all dof to zero on bottom line
LOAD = [[rept, [[1, 0.1], ]], ]     # setting y-dof a traction force

u, r = m.SolveElastic(K, BC, LOAD)

# %% Post process

m.Plot(u, alpha=0.2)
m.Plot(u, 50)

m.PlotContourStrain(u, cmap='RdBu')

m.PlotContourDispl(u, s=30)

m.PlotContourStress(u, hooke)

# possibility to plot directly as gauss points (without nodal averaging)
EN, ES = m.StrainAtGP(u)
gpfield = EN[:, 0]
vmax = np.max(abs(gpfield))
plt.figure()
plt.scatter(m.pgx, m.pgy, c=gpfield, cmap="RdBu", s=1, vmin=-vmax, vmax=vmax)
plt.colorbar()
plt.axis('off')
plt.axis('equal')
m.Plot(alpha=0.1)

m.VTKSol('test_displ', u)
