#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method 
    JC Passieux, INSA Toulouse, 2021       """

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg
import scipy as sp
import pyxel as px

#%% ============================================================================
''' Images '''
# ==============================================================================

# image numbers
imnums=np.array([53,54,57,58,61,62,65,66,69,70,75])
# image basename
testcase='dic_composite'
# generic image filename
imagefile=os.path.join('data',testcase,'zoom-0%03d_1.tif')
# constructing reference image filename (first image)
imref = imagefile % imnums[0]

# loading reference image with pyxel
f=px.Image(imref).Load()
# Plot reference image
f.Plot()

# Loading deformed image
imdef = imagefile % imnums[-2]
g = px.Image(imdef).Load()

#%% ============================================================================
''' Mesh '''
# ==============================================================================

# ABAQUS Mesh filename
meshfile=os.path.join('data',testcase,'abaqus_q4_m.inp')
# loading mesh with pyxel
m=px.ReadMeshINP(meshfile)
# Visualization of the mesh alone
plt.figure()
m.Plot()

#%% ============================================================================
''' Camera model '''
# ==============================================================================
# Calibration of the camera model
# cam=px.MeshCalibration(f,m,[1,2])
cam=px.Camera(np.array([ 1.05168768e+04,  5.13737634e-02, -9.65935782e-02, -2.65443047e-03]))

# Plot Mesh on the reference image
px.PlotMeshImage(f,m,cam)


#%% ============================================================================
''' Pre-processing  '''
# ==============================================================================

# Build the connectivity table
m.Connectivity()
# Build the quadrature rule; compute FE basis functions and derivatives
m.DICIntegration(cam)

# Multiscale initialization of the displacement
U0=px.MultiscaleInit(f,g,m,cam,scales=[3,2,1])
m.Plot(U0,30)

#%% ============================================================================
# Classic Modified Gauss Newton  ===============================================
# ==============================================================================
U=U0.copy()
# initialization of DIC engine
dic=px.DICEngine()
# Computing the DIC Hessian Operator
H=dic.ComputeLHS(f,m,cam)
# Factorization
H_LU=splalg.splu(H)
for ik in range(0,30):
    # Computing the DIC right hand side
    [b,res]=dic.ComputeRHS(g,m,cam,U)
    # System resolution
    dU=H_LU.solve(b)
    # update displacement
    U+=dU
    # estimate stagnation error
    err=np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc=%2.2f gl | dU/U=%1.2e" % (ik+1,np.std(res),err))
    if err<1e-3:
        break

# or equivalently in a compact form
U1,r=px.Correlate(f,g,m,cam,U=U0)

#%% ============================================================================
''' Post-processing '''
# ==============================================================================
# Visualization: Scaled deformation of the mesh
m.Plot(edgecolor='#CCCCCC')
m.Plot(U,30)
# Visualization: displacement fields
m.PlotContourDispl(U)
# Visualization: strain fields
m.PlotContourStrain(U)
# Plot deformed Mesh on deformed state image
px.PlotMeshImage(g,m,cam,U)

# Plot residual
m.PlotResidualMap(f,g,cam,U,1e5)

# Export for Paraview
m.VTKSol('dic_composite/Sol',U)
m.VTKIntegrationPoints(cam,f,g,U,filename='dic_composite/IntPts',iscale=0)

'''==========='''
''' EXERCISES '''
'''==========='''

#%% 1. Time Resolved DIC  ===================================================
UU=np.zeros((m.ndof,len(imnums)))

#...

m.VTKSolSeries('dic_composite/Sol',UU)

#%% 2. Tikhonov Regularization  =====================================================

A=m.Tikhonov()
a=100000
Ut=U0.copy()

#...

# Displacement field visualization using matplotlib
m.Plot(U,30)
m.Plot(Ut,30,edgecolor='r')

#%% 3. Mechanical Regularization  ===================================================

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
# repb1=m.SelectLine()
# repb2=m.SelectLine()
repb1=np.array([ 12,  13, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235])
repb2=np.array([ 14,  15, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268])
repb=np.append(repb1,repb2)
dofb=m.conn[repb,:].ravel()
D=np.ones_like(U)
D[dofb]=0
D=sp.sparse.diags(D)
A=K.T@D@K


U=U0.copy()
a=0.00001
Um=U0.copy()

#...

# Displacement field visualization using matplotlib
m.Plot(U,30)
m.Plot(Ut,30,edgecolor='r')

#%% 4. Experimental displacement driven simulation


#%% 5. Model Validation


#%% 6 Constitutive parameter identification


