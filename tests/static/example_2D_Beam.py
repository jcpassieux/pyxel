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
m.Plot()
repu, = np.where(m.n[:, 1] < -0.03499)
plt.plot(m.n[repu, 0], m.n[repu, 1], 'ro')
repu = m.conn[repu, :]

repf, = np.where(m.n[:, 1] > 0.03499)
plt.quiver(m.n[repf, 0], m.n[repf, 1], m.n[repf, 1]*1, m.n[repf, 1]*0, color='g')
repf = m.conn[repf, 0]

U = np.zeros(m.ndof)
F = np.zeros(m.ndof)
F[repf] = 100

# Solve
rep = np.setdiff1d(np.arange(m.ndof), repu)
repk = np.ix_(rep, rep)
F -= K@U

KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

# Plot results
plt.figure()
m.Plot(U, alpha=0.2)
m.Plot(U, 1000)
