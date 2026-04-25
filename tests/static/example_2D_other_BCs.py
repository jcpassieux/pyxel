# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:25:49 2026

Other kind of BC

@author: passieux
"""

import os
import numpy as np
import pyxel as px


# %% Testcase with 2D plane stress elasticity

fn = os.path.join('tri3.msh')
m = px.ReadMesh(fn)
m.KeepSurfElems()
m.Connectivity()

C = px.Hooke([200, 0.3], 'isotropic_2D_ps')  # Plane stress

m.GaussIntegration()
K = m.Stiffness(C)

# %%  FORCE AT THE TOP :


"""
distributed = False
means you prescribe concentrated force on each nodes
here 0.01 on each vertical dof
"""

repb = m.SelectEndLine('bottom', 1e-5, plot=False)
rept = m.SelectEndLine('top', 1e-5, plot=False)

BC = [[repb, [[0, 0], [1, 0]], ], ]   # setting all dof to zero on bottom line

LOAD = [[rept, [[1, 0.01], ], ], ]       # applying vertical force on top line

U, RF = m.SolveElastic(K, BC, LOAD, distributed=False)

m.Plot(alpha=0.1)
m.Plot(U, 1)

# %%


"""
Possibility to proceed step by step
"""
Fext = m.ApplyNeumann(LOAD, False)
Kd, Fd, Ud = m.ApplyDirichlet(K, BC)
U = m.LinearSolver(Kd, Fext, Fd)
RF = K@U-Fext

# %% FORCE AT THE TOP :


"""
distributed = True
means you apply a surface traction on the edge element
computes the quadrature of the surface force on the boundary mesh
"""

repb = m.SelectEndLine('bottom', 1e-5, plot=False)
rept = m.SelectEndLine('top', 1e-5, plot=False)

BC = [[repb, [[0, 0], [1, 0]], ], ]   # setting all dof to zero on bottom line

LOAD = [[rept, [[1, 1.25], ], ], ]   # applying vertical force on top line

U, RF = m.SolveElastic(K, BC, LOAD, distributed=True)

m.Plot(alpha=0.1)
m.Plot(U, 1)

# %%

# Different BC application methods :

U1, _ = m.SolveElastic(K, BC, LOAD, meth='penalty')
U2, _ = m.SolveElastic(K, BC, LOAD, meth='subs')
U3, _ = m.SolveElastic(K, BC, LOAD, meth='lagrange')

m.Plot(alpha=0.1)
m.Plot(U1, 1)
m.Plot(U2, 1)
m.Plot(U3, 1)
