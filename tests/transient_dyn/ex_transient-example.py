#!/usr/bin/env python
# coding: utf-8

# ## Importing packages
import numpy as np
from scipy.sparse import linalg as spl
import matplotlib.pyplot as plt

import pyxel as px


# %%
# ### Defining the material

# ## Definition of parameters
E = 210000                # Young modulus
nu = 0.3                  # Poisson ratio
rho = 7800e-9             # Volumic mass

# ## Proportional coefficients for damping
a = 2e-4
b = 1.

# ## Plane stress model
hooke = px.Hooke([E, nu], typc='isotropic_2D_ps')

# %%
# ### Loading the mesh
mesh = px.ReadMesh('support-1.msh')
mesh.KeepSurfElems()  # keep only cells of type quad or triangles.

# mesh.Plot()

# ### Computing the mass and stiffness matrices
mesh.Connectivity()                    # Computation of the connectivity
mesh.GaussIntegration()
K = mesh.Stiffness(hooke)              # Assembling of the stiffness matrix
M = mesh.Mass(rho)                     # Assembling of the mass matrix

# %%
# ### Elimination of boundary conditions

# ## Loading the numbering of the blocked nodes.
bc = mesh.SelectEndLine('left')
BC = [[bc, [[0, 0], [1, 0]]]]
K, _, _ = mesh.ApplyDirichlet(K, BC)

# ## Damping matrix
C = a*K + b*M

# %%
# ## Number of modes to compute
n = 25

# ## Solving the eigenvalue problem
valp, Q = spl.eigsh(K, n, M, 0., 'LM')

# ## Natural angular frequencies
wi = np.sqrt(np.abs(valp))
# ## Modal damping factor
xi = a*wi/2 + b/(2*wi)

print('The natural angular frequencies are: {}'.format(wi))
print('The modal damping factors are: {}'.format(xi))

# ## Adding the blocked value for the modal shape in order to plot it further

# mesh.Plot(Q[:,1],1)
mesh.PlotContourDispl(Q[:, 1], s=1)
# mesh.PlotContourStrain(Q[:,1], s=1)

# %%
# ## Numbering of the node for the FRF
nodeFRF = 293
dofFRF = (mesh.conn[nodeFRF, 1], mesh.conn[nodeFRF, 1])

nmodes = 10
# Truncated modal basis
Qn = Q[:, :nmodes]

# ## Modal matrices
mm = np.diag((Q.T @ M @ Q).diagonal())
kk = np.diag((Q.T @ K @ Q).diagonal())
cc = a*kk + b*mm

Omega = np.logspace(2, 4, 200)

# ## You can use a for loop or whatever the solution to compute the FRF
Hdamped = [abs(Q.dot(np.linalg.inv(kk+omega*cc*1j-omega**2*mm).dot(Q.T))[dofFRF]) for omega in Omega]
Hundamped = [abs(Q.dot(np.linalg.inv(kk-omega**2*mm).dot(Q.T))[dofFRF]) for omega in Omega]

plt.figure()
plt.loglog(Omega, Hdamped)
plt.loglog(Omega, Hundamped)

# %%
# ## Initial static solution
# Loading the numbering of the node where is applied the force
circleNode = mesh.SelectCircle([100, 50], 5)

# Creation of the right-hand-side F
LOAD = [[circleNode, [[0, 10], [1, 10]]]]
F = mesh.ApplyNeumann(LOAD)

# Solving the static problem Ku=F
q0 = mesh.LinearSolver(K, F)

qp0 = np.zeros(mesh.ndof)

# %%
T = 0.5


def analytical(h, q0, qp0):
    nt = int(T/h)
    tt = np.linspace(0, T, nt)
    p0 = Qn.T @ M @ q0
    pp0 = Qn.T @ M @ qp0
    # ## Initiating the list of p_i
    pi = []
    # ## Loop over the p_i
    for ii in range(nmodes):
        # ## Computing alpha and beta
        wd = wi[ii] * np.sqrt(np.abs(1-xi[ii]**2))
        alpha = p0[ii]
        beta = (pp0[ii]+xi[ii]*wi[ii]*p0[ii])/(wd)
        # ## Test if xi < 1 and computing the solution p_i for all t
        if xi[ii] < 1:
            pi.append(np.exp(-xi[ii]*wi[ii]*tt)*(alpha*np.cos(wd*tt)+beta*np.sin(wd*tt)))
        else:
            pi.append((alpha+beta)/2*np.exp((-xi[ii]*wi[ii]+wi[ii]*np.sqrt(xi[ii]-1))*tt)+(alpha-beta)/2*np.exp((-xi[ii]*wi[ii]-wi[ii]*np.sqrt(xi[ii]-1))*tt))
    # ## Vertical stack of the p_i
    p = np.vstack(pi)
    # ## Computing q
    q = Qn @ p
    return tt, q


def newmark(h, q0, qp0, beta, gamma):
    nt = int(T/h)
    tt = np.linspace(0, T, nt)
    q = [q0]
    qp = [qp0]
    qpp = [-C @ qp[-1] - K @ q[-1]]
    Mnew = spl.splu((M + C*h*gamma + K*beta*h**2))
    for t in tt[1:]:
        dn = q[-1] + h*qp[-1] + 0.5*h**2*(1-2*beta)*qpp[-1]
        vn = qp[-1] + h*(1-gamma)*qpp[-1]
        qpp.append(Mnew.solve(-K @ dn - C @ vn))
        q.append(dn + beta*h**2*qpp[-1])
        qp.append(vn + gamma*h*qpp[-1])
    return tt, np.vstack(q).T


h = 0.001
tana, qana = analytical(h, q0, qp0)
tnew, qnew = newmark(h, q0, qp0, 0.25, 0.5)


# %%
# plt.figure()
# plt.plot(tana,qana[mesh.conn[nodeFRF,1]])
# plt.plot(tnew,qnew[mesh.conn[nodeFRF,1]])


# %%
# ## Animate the response
anim = mesh.AnimatedPlot([qana, qnew], 50)
plt.show()
