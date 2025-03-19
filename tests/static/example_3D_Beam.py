# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 09:07:44 2024

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px
import scipy.sparse.linalg as splalg

# build a beam mesh from the composition of the unit cell by a macro mesh
box = np.array([[0, 0, 0], [0.1, 0.1, 0.1]])
macro = px.StructuredMeshHex8(box, 0.02)
micro = px.ReadMesh('3d_wire_full.inp', 3)
m = macro.FEComposition(micro)
m.Plot()

# Build FE operators
m.Connectivity(dof_per_node=6)
r = 0.001
d = 2*r
# Set beam properties [E, nu, S, Igz, k]
beam_prop = px.BeamProperties([200e9, 0.3, np.pi*r**2, np.pi*d**4/64, None, np.pi*d**4/64, np.pi*d**4/32])
K = m.StiffnessBeam(beam_prop)

# Apply BCs
repu = m.SelectEndLine('left')
BC = [[repu, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]], ]

repf = m.SelectEndLine('right')
LOAD = [[repf, [[1, 30], ]], ]

# Solve
U, RF = m.SolveElastic(K, BC, LOAD, distributed=False)

# Plot solution
m.Plot(U, 1000)


# %%

fn = 'qua4.msh'
macro = px.ReadMesh(fn)
macro.KeepSurfElems()
box = np.array([[-1, -1], [1, 1]])
micro = px.StructuredMeshQ4(box, 2)
micro = micro.BuildBoundaryMesh()
m = macro.FEComposition(micro)
m.n = np.c_[m.n, np.zeros(len(m.n))]
m.dim = 3
m.Plot()

m.Connectivity(dof_per_node=6)
r = 0.005
d = 2*r
# Set beam properties [E, nu, S, Igz, k]
beam_prop = px.BeamProperties([200e9, 0.3, np.pi*r**2, np.pi*d**4/64, None, np.pi*d**4/64, np.pi*d**4/32])
K = m.StiffnessBeam(beam_prop)

repu = m.SelectEndLine('bottom')
BC = [[repu, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]], ]

repfn = m.SelectEndLine('top')
repf = m.conn[repfn, 2]
F = np.zeros(m.ndof)
F[repf] = 3000 * m.n[repfn, 0] + 10

U, RF = m.SolveElastic(K, BC, F)

m.Plot(U, 1000)
