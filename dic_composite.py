#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2017       """

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg
import scipy as sp
import pyxel as px
import os

#%% ============================================================================
# Datafiles, images  ===========================================================
# ==============================================================================

imnums=np.array([53,54,57,58,61,62,65,66,69,70,75])
imagefile=os.path.join('data','dic_composite','zoom-0%03d_1.tif')
imref = imagefile % imnums[0]
f=px.Image(imref).Load()
f.Show()

#%% ============================================================================
# Mesh and Camera model  =======================================================
# ==============================================================================

''' Example1: Triangular GMSH Mesh in milimeter '''
meshfile=os.path.join('data','dic_composite','gmsh_t3_mm.msh')
m=px.ReadMeshGMSH(meshfile)
#cam=px.MeshCalibration(f,m,[1,2])
cam=px.Camera(np.array([10.49882137, 51.07774613, -6.53402857,  1.56607077]))
l0=1     # regularization length

''' Example2: Quadrilateral ABAQUS Mesh in meter '''
meshfile=os.path.join('data','dic_composite','abaqus_q4_m.inp')
m=px.ReadMeshINP(meshfile)
#cam=px.MeshCalibration(f,m,[1,2])
cam=px.Camera(np.array([ 1.05168768e+04,  5.13737634e-02, -9.65935782e-02, -2.65443047e-03]))
l0=0.005     # regularization length

''' Example3: Structured Built-in Quad Mesh in Pixels '''
box=np.array([[538,50],[638,950]])+0.5
N=np.array([10,10])
m=px.StructuredMeshQ4(box,N)
cam=px.Camera(np.array([1.,0.,-f.pix.shape[0],0.]))
l0=15     # regularization length

''' Example4: Structured Built-in Triangle Mesh in Pixels '''
box=np.array([[538,50],[638,950]])+0.5
N=np.array([10,10])
m=px.StructuredMeshT3(box,N)
cam=px.Camera(np.array([1.,0.,-f.pix.shape[0],0.]))
l0=15     # regularization length

# Plot Mesh on the reference image
px.PlotMeshImage(f,m,cam)
# Visualization of the mesh alone using Matplotlib
m.Plot()
# Or Visualization of the mesh using Paraview
m.VTKMesh(os.path.join('dic_composite','Mesh'))

#%% ============================================================================
# Pre-processing  ==============================================================
# ==============================================================================

# Build the connectivity table
m.Connectivity()
# Build the quadrature rule; compute FE basis functions and derivatives
m.DICIntegration(cam)

# Open reference image
imdef = imagefile % imnums[-2]
g = px.Image(imdef).Load()

# Multiscale initialization of the displacement
U0=px.MultiscaleInit(m,f,g,cam,3)
m.Plot(U0,30)

#%% ============================================================================
# Classic Modified Gauss Newton  ===============================================
# ==============================================================================
U=U0.copy()
dic=px.DICEngine()
H=dic.ComputeLHS(f,m,cam)
H_LU=splalg.splu(H)
for ik in range(0,30):
    [b,res]=dic.ComputeRHS(g,m,cam,U)
    dU=H_LU.solve(b)
    U+=dU
    err=np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/dic.dyn*100,err))
    if err<1e-3:
        break

# Visualization using matplotlib: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U,30)
# Visualization using matplotlib: displacement fields
m.PlotContourDispl(U)
# Visualization using matplotlib: strain fields
m.PlotContourStrain(U)
# or Displacement field visualization using Paraview
m.VTKSol(os.path.join('dic_composite','Sol_std'),U)
# Plot deformed Mesh on deformed state image
px.PlotMeshImage(g,m,cam,U)

#%% ============================================================================
# Tikhonov Regularization  =====================================================

V=np.zeros_like(U)
V[m.conn[:,0]]=np.cos(m.n[:,1]/l0*2*np.pi)
#m.PlotU(V,0.001)
#m.VTKSol(V,'PlaneWave')

H0=V.dot(H.dot(V))
#b,res=dic.ComputeRHS(g,m,cam,V)
#H0=res.T.dot(res)

L=m.Tikhonov()
L0=V.dot(L.dot(V))
l=H0/L0
U=U0.copy()

H_LU=splalg.splu(H+l*L)
for ik in range(0,30):
    [b,res]=dic.ComputeRHS(g,m,cam,U)
    dU=H_LU.solve(b-l*L.dot(U))
    U+=dU
    err=np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/dic.dyn*100,err))
    if err<1e-3:
        break


# Displacement field visualization using matplotlib
m.Plot(U,30)
# or Displacement field visualization using Paraview
m.VTKSol(os.path.join('dic_composite','Sol_Tikhonov'),U)


#%% ============================================================================
# Mechanical Regularization  ===================================================
# ==============================================================================

El=1.0e+10
Et=1.9e+10
vtl=0.18
Glt=1.0e+09
vlt=vtl*El/Et
alp=1/(1-vlt*vtl)
hooke=np.array([[alp*El     , alp*vtl*El,0    ],
                [alp*vlt*Et , alp*Et    ,0    ],
                [0          , 0         ,2*Glt]])

K=m.Stiffness(hooke)
# Select the non-free boundary nodes (here two lines)
# repb1=px.SelectMeshLine(m)
# repb2=px.SelectMeshLine(m)
repb1=np.array([ 12,  13, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235])
repb2=np.array([ 14,  15, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268])
repb=np.append(repb1,repb2)
dofb=m.conn[repb,:].ravel()
D=np.ones_like(U)
D[dofb]=0
D=sp.sparse.diags(D)
KK=K.T.dot(D.dot(K))

V=np.zeros_like(U)
V[m.conn[:,0]]=np.cos(m.n[:,1]/l0*2*np.pi)
H0=V.dot(H.dot(V))
K0=V.dot(KK.dot(V))
l=H0/K0


H_LU=splalg.splu(H+l*KK)
for ik in range(0,30):
    [b,res]=dic.ComputeRHS(g,m,cam,U)
    dU=H_LU.solve(b-l*KK.dot(U))
    U+=dU
    err=np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/dic.dyn*100,err))
    if err<1e-3:
        break

# Displacement field visualization using matplotlib
m.Plot(U,30)
# or Displacement field visualization using Paraview
m.VTKSol(os.path.join('dic_composite','Sol_EquilibriumGap'),U)


#%% ============================================================================
# EXAMPLE: Time resolved DIC Code with Tikhonov regularization ==================
# ==============================================================================

imdef=imagefile % imnums[1]
g=px.Image(imdef).Load()

V=np.zeros_like(U)
V[m.conn[:,0]]=np.cos(m.n[:,1]/l0*2*np.pi)
H0=V.dot(H.dot(V))

L=m.Tikhonov()
L0=V.dot(L.dot(V))
l=H0/L0

U=np.zeros(m.ndof)
H_LU=splalg.splu(H+l*L)

m.PVDFile(os.path.join('dic_composite','Sol'),'vtu',1,len(imnums))
m.VTKSol(os.path.join('dic_composite','Sol_0_0'),np.zeros(m.ndof))
for ig in range(1,len(imnums)):
    imdef=imagefile % imnums[ig]
    g=px.Image(imdef).Load()
    for ik in range(0,30):
        [b,res]=dic.ComputeRHS(g,m,cam,U)
        dU=H_LU.solve(b-l*L.dot(U))
        U+=dU
        err=np.linalg.norm(dU)/np.linalg.norm(U)
        print("Iter # %2d | disc/dyn=%2.2f %% | dU/U=%1.2e" % (ik+1,np.std(res)/dic.dyn*100,err))
        if err<1e-3:
            break
    # Deformed mesh output
    m.Plot(edgecolor='#CCCCCC')
    m.Plot(U,30)
    plt.savefig(os.path.join('vtk','dic_composite','traction_%02d.png' % ig))
    plt.close()

    # Mesh on Image output    
    px.PlotMeshImage(g,m,cam,U)
    plt.savefig(os.path.join('vtk','dic_composite','MshImg_%02d.png' % ig))
    plt.close()
    
    # Paraview Output
    m.VTKSol(os.path.join('dic_composite','Sol_0_%d' % ig),U)
