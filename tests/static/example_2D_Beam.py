# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:00:39 2024

@author: passieux
"""

import numpy as np
import matplotlib.pyplot as plt
import pyxel as px
import scipy.sparse.linalg as splalg


# build a beam mesh from the edges of a 2D mesh of linear quad elements
fn = 'qua4.msh'
macro = px.ReadMesh(fn)
macro.KeepSurfElems()
box = np.array([[-1, -1], [1, 1]])
micro = px.StructuredMeshQ4(box, 2)
micro = micro.BuildBoundaryMesh()
m = macro.FEComposition(micro)
m.Plot()

# Build FE operators
m.Connectivity(dof_per_node=3)
r = 0.005
d = 2*r
# Set beam properties [E, nu, S, Igz, k]
beam_prop = px.BeamProperties([200e9, 0.3, np.pi*r**2, np.pi*d**4/64, None])
K = m.StiffnessBeam(beam_prop)

# Apply BCs
repu = m.SelectEndLine('bottom', 1e-5)
repf = m.SelectEndLine('top', 1e-5)

BC = [[repu, [[0, 0], [1, 0], [2, 0]]]]    # setting all dofs to 0
LOAD = [[repf, [[0, 100],]]]     # prescribing x-dimerction force.

u, r = m.SolveElastic(K, BC, LOAD)

# Plot results
plt.figure()
m.Plot(u, alpha=0.2)
m.Plot(u, 5e4)

