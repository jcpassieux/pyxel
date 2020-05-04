#!/usr/bin/env python
# coding: utf-8

### Importing packages
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg as spl
import matplotlib.pyplot as plt
import matplotlib.collections as cols
import time

import sys
sys.path.append('../..')
import pyxel as px
import matplotlib.animation as animation


#%%
#### Defining the material

### Definition of parameters
E = 210000                # Young modulus
nu = 0.3                  # Poisson ratio
rho = 7800e-9             # Volumic mass

### Proportional coefficients for damping
a = 2e-4
b = 1.

### Plane stress model
hooke = E / (1-nu**2) * np.array([[1,nu,0],[nu,1,0],[0,0,0.5*(1-nu)]]) 

#%%
#### Loading the mesh
mesh = px.ReadMeshGMSH('support-1.msh')
#mesh.Plot()


#### Computing the mass and stiffness matrices
mesh.Connectivity()                            # Computation of the connectivity
K = mesh.Stiffness(hooke).tocsc()              # Assembling of the stiffness matrix
M = mesh.Mass(rho).tocsc()                     # Assembling of the mass matrix

#%%
#### Elimination of boundary conditions

### Loading the numbering of the blocked nodes. 
bc = np.loadtxt('bc-1.txt').astype(int)
### Obtaining the numbering of the eliminated dofs
dofBC = np.sort(mesh.conn[bc,:].ravel())
### Numbering of the nodes to keep
nodeToKeep = np.delete(np.arange(mesh.conn.shape[0]),bc)
### Numbering of the nodes to keep
dofToKeep = np.delete(np.arange(mesh.ndof),dofBC)
dofToKeepMat = np.ix_(dofToKeep,dofToKeep)

### Computing the new connectivity of dofs
I = np.argsort(mesh.conn[nodeToKeep,:].ravel())
II = np.argsort(I)
newConn = np.arange(len(dofToKeep))[II].reshape((dofToKeep.shape[0]//2,mesh.dim))

#%%
### Matrices with the eliminated dofs
Kbc = K[dofToKeepMat]
Mbc = M[dofToKeepMat]
Cbc = a*Kbc + b*Mbc

#%%
### Number of modes to compute
n = 25

### Solving the eigenvalue problem
valp,Qbc=spl.eigsh(Kbc, n, Mbc,0.,'LM')

### Natural angular frequencies
wi = np.sqrt(np.abs(valp))
### Modal damping factor
xi = a*wi/2 + b/(2*wi)

print('The natural angular frequencies are: {}'.format(wi))
print('The modal damping factors are: {}'.format(xi))

### Adding the blocked value for the modal shape in order to plot it further
Q = np.zeros((mesh.ndof,Qbc.shape[1]))
Q[dofToKeep,:] = Qbc

#%%
### Numbering of the node for the FRF
nodeFRF = 293
dofFRF = (newConn[nodeFRF,1],newConn[nodeFRF,1])

nmodes = 10
# Truncated modal basis
Qn = Q[:,:nmodes]
Qnbc = Qbc[:,:nmodes]

### Modal matrices
mm = np.diag(Qnbc.T.dot(Mbc.dot(Qnbc)).diagonal())
kk = np.diag(Qnbc.T.dot(Kbc.dot(Qnbc)).diagonal())
cc = a*kk + b*mm

Omega = np.logspace(2,4,200)

### You can use a for loop or whatever the solution to compute the FRF
Hdamped = [abs(Qnbc.dot(np.linalg.inv(kk+omega*cc*1j-omega**2*mm).dot(Qnbc.T))[dofFRF]) for omega in Omega]
Hundamped = [abs(Qnbc.dot(np.linalg.inv(kk-omega**2*mm).dot(Qnbc.T))[dofFRF]) for omega in Omega]

plt.figure()
plt.loglog(Omega,Hdamped)
plt.loglog(Omega,Hundamped)

#%%
### Initial static solution
# Loading the numbering of the node where is applied the force
circleNode = np.loadtxt('circle.txt').astype(int)
# Obtaining the numbering of the dofs
circleDof = newConn[circleNode,:].ravel()

# Creation of the right-hand-side F
Fd = np.zeros(mesh.ndof)
Fd[mesh.conn[circleNode,:].ravel()] = 10*np.ones(len(circleDof))
F = Fd[dofToKeep]

# Solving the static problem Ku=F
u = spl.spsolve(Kbc,F)

# Adding the blocked dof to the result
ustat = np.zeros(mesh.ndof)
ustat[dofToKeep] = u
q0 = ustat
qp0 = np.zeros(mesh.ndof)

#%%
T = 0.5

def analytical(h,q0,qp0):
    nt = int(T/h)
    tt = np.linspace(0,T,nt)
    p0 = Qn.T.dot(M.dot(q0))
    pp0 = Qn.T.dot(M.dot(qp0))
    ### Initiating the list of p_i
    pi = []
    ### Loop over the p_i
    for ii in range(nmodes):
       ### Computing alpha and beta
        wd = wi[ii] * np.sqrt(np.abs(1-xi[ii]**2))
        alpha = p0[ii]
        beta = (pp0[ii]+xi[ii]*wi[ii]*p0[ii])/(wd)
        ### Test if xi < 1 and computing the solution p_i for all t
        if xi[ii] < 1:
            pi.append(np.exp(-xi[ii]*wi[ii]*tt)*(alpha*np.cos(wd*tt)+beta*np.sin(wd*tt)))
        else:
            pi.append((alpha+beta)/2*np.exp((-xi[ii]*wi[ii]+wi[ii]*np.sqrt(xi[ii]-1))*tt)+(alpha-beta)/2*np.exp((-xi[ii]*wi[ii]-wi[ii]*np.sqrt(xi[ii]-1))*tt))
    
    ### Vertical stack of the p_i
    p = np.vstack(pi)
    ### Computing q
    q = Qn.dot(p)
    return tt,q
    
def newmark(h,q0,qp0,beta,gamma):
    nt = int(T/h)
    tt = np.linspace(0,T,nt)
    q = [q0]
    qp = [qp0]
    qpp = [-Cbc.dot(qp[-1]) -Kbc.dot(q[-1])]
    Mnew = spl.splu((Mbc+Cbc*h*gamma+Kbc*beta*h**2))
    for t in tt[1:]:
        dn = q[-1]+h*qp[-1]+0.5*h**2*(1-2*beta)*qpp[-1]
        vn = qp[-1]+h*(1-gamma)*qpp[-1]
        qpp.append(Mnew.solve(-Kbc.dot(dn)-Cbc.dot(vn)))
        q.append(dn+beta*h**2*qpp[-1])
        qp.append(vn+gamma*h*qpp[-1])
    return tt,np.vstack(q).T


h = 0.001
tana,qana = analytical(h,q0,qp0)
tnew,new = newmark(h,u,np.zeros(u.shape),0.25,0.5)
qnew = np.zeros((mesh.ndof,new.shape[1]))
qnew[dofToKeep,:] = new

#%%
#plt.figure()
#plt.plot(tana,qana[mesh.conn[nodeFRF,1]])
#plt.plot(tnew,qnew[mesh.conn[nodeFRF,1]])


#%%
### Animate the response
anim = mesh.AnimatedPlot([qana,qnew],50)
plt.show()