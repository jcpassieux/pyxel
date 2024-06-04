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
m.Plot()
repu, = np.where(m.n[:, 0] < 1e-5)
plt.plot(m.n[repu, 0], m.n[repu, 1], m.n[repu, 2], 'ro')
repu = m.conn[repu, :]

repf, = np.where(m.n[:, 0] > 0.1-1e-5)
plt.plot(m.n[repf, 0], m.n[repf, 1], m.n[repf, 2], 'go')
repf = m.conn[repf, 1]

U = np.zeros(m.ndof)
F = np.zeros(m.ndof)
F[repf] = 30

rep = np.setdiff1d(np.arange(m.ndof), repu)
repk = np.ix_(rep, rep)
F -= K@U

# Solve
KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

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

m.Plot()
repu, = np.where(m.n[:, 1] < -0.03499)
plt.plot(m.n[repu, 0], m.n[repu, 1], m.n[repu, 2], 'ro')
repu = m.conn[repu, :]

repfn, = np.where(m.n[:, 1] > 0.03499)
plt.plot(m.n[repfn, 0], m.n[repfn, 1], m.n[repfn, 2], 'go')
repf = m.conn[repfn, 2]

U = np.zeros(m.ndof)
F = np.zeros(m.ndof)
F[repf] = 3000 * m.n[repfn, 0] + 10

rep = np.setdiff1d(np.arange(m.ndof), repu)
repk = np.ix_(rep, rep)
F -= K@U

KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

m.Plot(U, 1000)
