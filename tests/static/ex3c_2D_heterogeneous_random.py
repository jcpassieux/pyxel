# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:28:27 2026

Example 3: a plate with different material parameters for each element

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px


box = np.array([[0, 0], [2, 1]])
m = px.StructuredMesh(box, 0.01)
m.Plot()

m.Connectivity()
m.GaussIntegration()

# number of quad4 elems from elem dict
nquad = len(m.e[3])

# build a elem-like dict of Young Modulus values
E_elem = {3: 1e4 + np.random.randn(nquad) * 1e3}
E_pg = m.Elem2GaussPoint(E_elem)
v_elem = {3: 0.3 + np.random.randn(nquad) * 0.05}
v_pg = m.Elem2GaussPoint(v_elem)

# build a (3 x 3 x npg) hooke array
hooke = E_pg / (1 - v_pg**2) * np.array([[v_pg**0, v_pg, 0*v_pg],
                                         [v_pg, v_pg**0, v_pg*0],
                                         [v_pg*0, v_pg*0, (1 - v_pg) / 2]])

K = m.Stiffness(hooke)

# Dirichlet BC at y = 0
repb = m.SelectEndLine('left', 1e-5)
rept = m.SelectEndLine('right', 1e-5)

BC = [[repb, [[0, 0], [1, 0]]], ]   # setting all dof to zero on bottom line
LOAD = [[rept, [[0, 0.1], ]], ]     # setting y-dof a traction force

u, r = m.SolveElastic(K, BC, LOAD)

# %% Post process

m.Plot(u, alpha=0.2)
m.Plot(u, 50)

m.PlotContourStrain(u, cmap='RdBu')

m.PlotContourDispl(u, s=30)

m.PlotContourStress(u, hooke)

# possibility to plot directly as gauss points (without nodal averaging)
EN, ES = m.StrainAtGP(u)
gpfield = EN[:, 1]
vmax = np.max(abs(gpfield))
plt.figure()
plt.scatter(m.pgx, m.pgy, c=gpfield, cmap="RdBu", s=1, vmin=-vmax, vmax=vmax)
plt.colorbar()
plt.axis('off')
plt.axis('equal')
m.Plot(alpha=0.1)

m.VTKSol('test_displ', u)
