# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 07:22:16 2026

Testcase with explicit nonlinear hooke matrix

@author: passieux
"""

import numpy as np
import sympy as sp
import pyxel as px

# %% Defining an arbitrary nonlinear hooke law


def NonLinearHooke(p, En, Es):
    exx = En[:, 0]
    eyy = En[:, 1]

    def fun(eyy, s):
        y = 1 - eyy/0.1 * s
        dy = 0*eyy - 1/0.1 * s
        return y, dy

    f, df = fun(eyy, p[4])
    h, dh = fun(eyy, p[5])

    # Hooke operator
    C = np.zeros((3, 3, len(eyy)))
    C[0, 0] = p[0]
    C[0, 1] = p[1] * h
    C[1, 0] = p[1] * h
    C[1, 1] = p[2] * f
    C[2, 2] = p[3]
    # tangent operator
    Ct = np.zeros((3, 3, len(eyy)))
    Ct[0, 0] = p[0]
    Ct[0, 1] = p[1] * (h + dh*eyy)
    Ct[1, 0] = p[1] * h
    Ct[1, 1] = p[1] * dh * exx + p[2] * (f + df*eyy)
    Ct[2, 2] = p[3]
    return C, Ct

# %%  MESH AND BC


box = np.array([[0, 0], [1, 1]])
m = px.OpenHolePlateUnstructured(box, 0.2, [0.5, 0.5], 0.05, 0.03)
m.Connectivity()
m.GaussIntegration()

# Dirichlet BC at y = -0.035
repu = m.SelectEndLine('bottom', plot=False)
BC = [[repu,      [[1, 0], ], ],
      [repu[[0]], [[0, 0], ], ], ]

# Neumann BC : distributed compression at the top
repf = m.SelectEndLine('top', plot=False)
LOAD = [[repf, [[1, -0.1], ], ], ]
Fext = m.ApplyNeumann(LOAD)

# Materials Parameters
E = 1.
v = 0.3
c00 = E / (1 - v**2)
c22 = E / (1 - v**2) * (1 - v) / 2
c01 = E / (1 - v**2) * v
c11 = E / (1 - v**2)
p = [c00, c01, c11, c22, 0.9, 0.9]

# %%  NEWTON


U = np.zeros(m.ndof)
dirichlet_dof = m.conn[repu, :].ravel()
for it in range(30):
    En, Es = m.StrainAtGP(U)
    Cnl, Ct = NonLinearHooke(p, En, Es)
    Sn, Ss = px.Strain2Stress(Cnl, En, Es)
    Fint = m.ComputeInternalForce(Sn, Ss)
    Kt = m.Stiffness(Ct)
    Ktd, Fd, _ = m.ApplyDirichlet(Kt, BC, 'penalty')
    R = Fext - Fint
    R[dirichlet_dof] = 0
    dU = m.LinearSolver(Ktd, R)
    U += dU
    res = np.linalg.norm(R) / np.linalg.norm(Fext)
    resu = np.linalg.norm(dU) / np.linalg.norm(U)
    print('Iter #%2d | res = %2.3e | dU/U = %2.3e' % (it, res, resu))
    if resu < 1e-10:
        break

# %% Post processing


m.Plot(alpha=0.2)
m.Plot(U, 1)

m.PlotContourStrain(U, s=1)
m.PlotContourStress(U, Cnl, s=1)

m.PlotContourDispl(U, s=1)

RF = Fint - Fext
m.PlotContourDispl(RF, s=0)


# %%


ex, ey, exy, c00, c01, c11, c22, p1, p2, p3, p4 = sp.symbols(
    'varepsilon_x, varepsilon_y, varepsilon_x_y, c_0_0, c_0_1, c_1_1, c_2_2, p_1, p_2, p_3, p_4')
f = sp.Function('f')(ey)
h = sp.Function('h')(ey)

eps = sp.Matrix([[ex], [ey], [2*exy]])

C = sp.Matrix([[c00, c01*h, 0],
               [c01*h, c11*f, 0],
               [0, 0, c22]])
sig = C*eps
Ct = sp.Matrix([sig.diff(ex).T, sig.diff(ey).T, sig.diff(exy).T/2]).T
