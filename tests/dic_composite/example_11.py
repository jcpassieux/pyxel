#!/usr/bin
# -*- coding: utf-8 -*-
""" Finite Element Digital Image Correlation method
    JC Passieux, INSA Toulouse, 2026

    Example 11 : Solving DIC in parallel on subdomains

"""

import pyxel as px
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as splalg

# %% IMAGES
f = px.Image('zoom-0053_1.tif').Load()
# f.Plot()
g = px.Image('zoom-0070_1.tif').Load()
# g.Plot()

# %% MESH from a region of interest
# f.SelectROI()
roi = np.array([[ 550.5,   50.5], [ 850.5,  450.5]])
mall, cam = px.MeshFromROI(roi, 20)
px.PlotMeshImage(f, mall, cam)

# %% Splitting the domain into subdomains
mall.Connectivity()

roi1 = np.array([[ 550,   50], [ 700,  250]])
roi2 = np.array([[ 550,   250], [ 700,  450]])
roi3 = np.array([[ 700,   50], [ 850,  250]])
roi4 = np.array([[ 700,   250], [ 850,  450]])
roi = [roi1, roi2, roi3, roi4]

m = []
color = ['r', 'b', 'k', 'c']
for i in range(4):
    mi = mall.Copy()
    mi.RemoveElemsOutsideRoi(roi[i], cam)
    m += [mi, ]
    mi.Plot(edgecolor=color[i])

# m[0].Plot()
# plt.plot(m[0].n[:, 0], m[0].n[:, 1], 'k.')

#%% Building H operator in parallel on the subdomains
dic = []
for i in range(4):
    m[i].DICIntegrationPixel(cam)
    dic += [px.DICEngine(), ]
    if i==0:
        H = dic[i].ComputeLHS(f, m[i], cam)
    else:
        H += dic[i].ComputeLHS(f, m[i], cam)   

H_LU = splalg.splu(H)

# px.PlotMeshImage(f, mall, cam)
# u, v = cam.P(m[0].pgx, m[0].pgy)
# plt.plot(u, v, 'y.')


# %% checking that parallel and monolithic operator are equal
# mall.DICIntegrationPixel(cam)
# dicall = px.DICEngine()
# Hall = dicall.ComputeLHS(f, mall, cam)  

# Hall = Hall.toarray()
# H = H.toarray()

# plt.imshow(Hall)
# plt.figure()
# plt.imshow(H)
# Hdiff = H-Hall

# %% Parallel (domain independent) Initialization

# averaging the interface dofs
scale = np.zeros(mall.ndof)
for i in range(len(roi)):
    scale[mall.conn[np.unique(m[i].e[3].ravel())].ravel()] += 1
# mall.PlotContourDispl(scale)

U0 = np.zeros(mall.ndof)
for i in range(len(roi)):
    Ui, UV = px.DISFlowInit(f, g, m[i], cam)
    # m[i].Plot(Ui, s=30)
    usddof = m[i].conn[np.unique(m[i].e[3].ravel()), :]
    U0[usddof] += Ui[usddof]

U0 = U0/scale
mall.Plot(U0, s=30)
    
# %% parallel GN iterations
U = U0.copy()
for k in range(30):
    b = np.zeros(m[0].ndof)
    res = np.zeros(0)
    for i in range(len(roi)):
        bi, resi = dic[i].ComputeRHS(g, m[i], cam, U)
        b += bi
        res = np.append(res, resi)
    dU = H_LU.solve(b)
    U += dU
    err = np.linalg.norm(dU) / np.linalg.norm(U)
    print("Iter # %2d | std(res)=%2.2f gl | dU/U=%1.2e" % (k+1, np.std(res), err))
    if err < 1e-3:
        break

mall.Plot(U, s=30)
