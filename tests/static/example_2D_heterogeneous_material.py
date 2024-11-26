# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:41:15 2024

@author: passieux
"""

import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg
import matplotlib.pyplot as plt

# %% Tiny testcase with two elements > handmade elem sets
# o  ----------------- ->
# o |        |        |->
# o |  hard  |  soft  |->
# o |        |        |->
# ///----------------- ->

box = np.array([[0, 0], [2, 1]])
m = px.StructuredMeshQ4(box, 1)
m.Plot()

# define cell_sets as a dict [sets] of dict [elemtype]
m.cell_sets = {'hard': {3: np.array([0])},
               'soft': {3: np.array([1])}}

m.Write('test.vtu')

m.Connectivity()
m.GaussIntegration()

C = dict()
C['soft'] = px.Hooke([1, 0.3], 'isotropic_2D_ps')
C['hard'] = px.Hooke([100, 0.3], 'isotropic_2D_ps')
hooke = m.AssignMaterial2GaussPoint(C)

K = m.Stiffness(hooke)

nodes_left = [0, 1]
nodes_left_bottom = [0, ]
BC = [[nodes_left, [[0, 0], ]],         # blocking x-dof for left nodes
      [nodes_left_bottom, [[1, 0],]]]   # blocking y-dof for node 0

nodes_right = [4, 5]
LOAD = [[nodes_right, [[0, 1], ]]]      # set unit force on x-dof

u, r = m.SolveElastic(K, BC, LOAD)

m.Plot(u, 1)

# %% Testcase *.inp mesh already including element sets
fn = 'rve_mesh.inp'
m = px.ReadMesh(fn)
m.KeepSurfElems()
m.Plot()

m.Write('test.vtu')
m.Connectivity()
m.GaussIntegration()

C = dict()
C['Set-1'] = px.Hooke([100, 0.3], 'isotropic_2D_ps')
C['Set-2'] = px.Hooke([1, 0.3], 'isotropic_2D_ps')
hooke = m.AssignMaterial2GaussPoint(C)

# hooke = px.Hooke([100, 0.3], 'isotropic_2D_ps')
K = m.Stiffness(hooke)

# Dirichlet BC at y = 0
repb = m.SelectEndLine('bottom', 1e-5)
rept = m.SelectEndLine('top', 1e-5)

BC = [[repb, [[0, 0], [1, 0]]],   # setting all dof to zero on bottom line
      [rept, [[1, 0.1], ]]]       # setting y-dof of top line to 0.1

u, r = m.SolveElastic(K, BC)
m.Plot(u, 50)

# %% Post-processing
m.Plot(u, alpha=0.2)
m.Plot(u, 50)

m.PlotContourStrain(u, cmap='RdBu')

m.PlotContourDispl(u, s=30)

m.PlotContourStress(u, hooke)

# possibility to plot directly as gauss points
EN, ES = m.StrainAtGP(u)
plt.scatter(m.pgx, m.pgy, c=EN[:, 1], cmap="RdBu", s=1)
plt.colorbar()
plt.axis('off')
plt.axis('equal')
m.Plot(alpha=0.1)

m.VTKSol('test_displ', u)
