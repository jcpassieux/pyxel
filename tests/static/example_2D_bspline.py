# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:14:37 2023

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px
import scipy.sparse as sps

# %% initial cad
a = 0.925
Xi = np.array([[0.5, 0.75, 1], 
               [0.5*a, 0.75*a, 1*a],
               [0, 0, 0]])
Yi = np.array([[0, 0, 0],
               [0.5*a, 0.75*a, 1*a], 
               [0.5, 0.75, 1]])
ctrlPts = np.array([Xi, Yi])
degree = [2, 2]
kv = np.array([0, 0, 0, 1, 1, 1])
knotVect = [kv, kv]

# %% mesh refinement
n = 10
newx = np.linspace(0, 1, n+2)[1:-1]
n = 5
newy = np.linspace(0, 1, n+2)[1:-1]
m = px.BSplinePatch(ctrlPts, degree, knotVect)
m.KnotInsertion([newx, newy])
# m.DegreeElevation(np.array([3, 3]))

m.Plot()

# %% Stiffness matrix
C = px.Hooke([1, 0.3])
m.Connectivity()
m.GaussIntegration()
K = m.Stiffness(C)

# %% Boundary conditions
# top_left = m.SelectLine()
top_left = np.array([12, 25, 38, 51, 64, 77, 90, 103], dtype=int)
# bot_right = m.SelectLine()
bot_right = np.array([0, 13, 26, 39, 52, 65, 78, 91], dtype=int)

dof_pin = m.conn[top_left, :].ravel()
dof = np.arange(m.ndof)
U = np.zeros(m.ndof)
F = U.copy()

F[m.conn[bot_right, 0]] = 1

rep = np.setdiff1d(dof, dof_pin)
repk = np.ix_(rep, rep)

# %% Linear system
U[rep] = sps.linalg.spsolve(K[repk], F[rep])

# %% Post-processing
m.Plot(alpha=0.5)
m.Plot(U=U*2e-4)

m.VTKSol('output', U)
