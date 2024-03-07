# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:41:15 2024

@author: passieux
"""

import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg

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

rep = np.array([2, 3, 4, 5, 7, 8, 9, 10, 11])
repk = np.ix_(rep, rep)
f = np.zeros(m.ndof)
f[4:6] = 1
u = np.zeros(m.ndof)
u[rep] = splalg.spsolve(K[repk], f[rep])

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
repu, = np.where(m.n[:, 1] < 1e-5)
repu = np.append(m.conn[repu[0], 0], m.conn[repu, 1])

# Dirichlet BC at y = 25 second dof only
repf, = np.where(m.n[:, 1] > 25-1e-5)
repf = m.conn[repf, 1]

U = np.zeros(m.ndof)
U[repf] = 0.1
U[repu] = 0.0

rep = np.setdiff1d(np.arange(m.ndof), np.append(repu, repf))
repk = np.ix_(rep, rep)
F = -K@U

KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

# %% Post-processing
m.Plot(U, alpha=0.2)
m.Plot(U, 50)

m.PlotContourStrain(U)

m.PlotContourDispl(U, s=30)

m.PlotContourStress(U, hooke)

m.VTKSol('test_displ', U)
