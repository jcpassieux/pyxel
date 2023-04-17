# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 08:16:42 2021

@author: passieux
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pyxel as px

INTERACTIVE = False
PLOT = True

#%% Test functions

def one_element_integration_test(m, cam):
    m.Connectivity()
    m.GaussIntegration()
    M0 = m.Mass(1).A
    K0 = m.Stiffness(px.Hooke([11,0.33])).A
    L0 = m.Tikhonov().A
    plt.figure()
    m.Plot()
    plt.plot(m.pgx,m.pgy,'k.')
    plt.title('GaussIntegration')
    
    m.DICIntegration(cam, G=True)
    M1=m.Mass(1).A
    K1=m.Stiffness(px.Hooke([11,0.33])).A
    L1 = m.Tikhonov().A
    plt.figure()
    m.Plot()
    plt.plot(m.pgx,m.pgy,'k.')
    plt.title('DICIntegration (optimised rule)')

    m.DICIntegrationFast(cam, G=True)
    M3=m.Mass(1).A
    K3=m.Stiffness(px.Hooke([11,0.33])).A
    L3 = m.Tikhonov().A
    plt.figure()
    m.Plot()
    plt.plot(m.pgx,m.pgy,'k.')
    plt.title('DICIntegration (Fast rule)')
    
    m.DICIntegrationPixel(cam)
    M2=m.Mass(1).A
    plt.figure()
    m.Plot()
    plt.plot(m.pgx,m.pgy,'k.')
    plt.title('Pixel Integration')
    
    print("MASS Gauss vs Opt %2f %%" % (np.linalg.norm(M0 - M1) / np.linalg.norm(M0) * 100, ) )
    print("MASS Gauss vs Pixel %2f %%" % (np.linalg.norm(M2 - M0) / np.linalg.norm(M0) * 100, ) )
    print("MASS Gauss vs Fast %2f %%" % (np.linalg.norm(M3 - M0) / np.linalg.norm(M0) * 100, ) )
    print("STIFFNESS Gauss vs Opt %2f %%" % (np.linalg.norm(K0 - K1) / np.linalg.norm(K0) * 100, ) )
    print("STIFFNESS Gauss vs Fast %2f %%" % (np.linalg.norm(K0 - K3) / np.linalg.norm(K0) * 100, ) )
    print("TIKO Gauss vs Opt %2f %%" % (np.linalg.norm(L0 - L1) / np.linalg.norm(L0) * 100, ) )
    print("TIKO Gauss vs Fast %2f %%" % (np.linalg.norm(L0 - L3) / np.linalg.norm(L0) * 100, ) )
    
def postpro_test(f, g, m, cam, U, res):
    if PLOT:
        plt.figure()
        m.Plot()
        plt.plot(m.pgx,m.pgy,'k.')
        plt.title('Mesh and Integration Points')
        
        plt.figure()
        m.Plot(edgecolor='#CCCCCC')
        m.Plot(U, 30, facecolor='r')
        px.PlotMeshImage(f, m, cam)
        px.PlotMeshImage(g, m, cam, U)
        
        m.PlotContourDispl(U, s=30)
        m.PlotContourStrain(U)
        m.PlotResidualMap(res, cam, 1e5)
    
    m.VTKMesh('unitary_test_0')
    m.VTKSol('unitary_test_1', U)
    
#%% Standard test Q4
imagefile = 'zoom-0%03d_1.tif'
imref = imagefile % 53
f = px.Image(imref).Load()
imdef = imagefile % 70
g = px.Image(imdef).Load()
m = px.ReadMeshINP('abaqus_q4_m.inp')
p = np.array([1.05449047e+04, 5.12335842e-02, -9.63541211e-02, -4.17489457e-03])
cam = px.Camera(p)

m.Connectivity()
conn = np.c_[np.arange(1500),np.arange(1500) + 1500]
res, _ = np.where(m.conn != conn)
if m.ndof != 3000 or len(res) != 0:
    print('************** PB in Connectivity')

m.GaussIntegration()
if m.npg != 5616 \
or int(np.sum(m.wdetJ) * 1e12) != 2623910771 \
or int(np.std(m.wdetJ) * 1e15) != 520884059:
    print('************** PB in GaussIntegration')

K = m.Stiffness(px.Hooke([11,0.33]))
if int(np.sum(K.diagonal()) * 1e10) != 664248104871388 \
or int(np.std(K.diagonal()) * 1e12) != 5645081239462:
    print('************** PB in Stiffness')

L = m.Tikhonov()
if int(np.sum(L.diagonal()) * 1e10) != 80614434627292 \
or int(np.std(L.diagonal()) * 1e12) != 502557306565:
    print('************** PB in Tikhonov')

M = m.Mass(1.8)
if int(np.sum(M.diagonal()) * 1e15) != 4198257234574 \
or int(np.std(M.diagonal()) * 1e15) != 1562119459:
    print('************** PB in Mass')

m.DICIntegrationPixel(cam)
X, Y = cam.P(m.pgx, m.pgy)
notint, = np.where((X - np.round(X) > 1e-12))
if len(notint) != 0:
    print('************** PB in DICIntegrationPixel or Camera')

m.DICIntegration(cam, G=True)
if int(m.dphixdx.diagonal().sum() * 1e10) != 1664277086951 \
or int(m.dphixdx.diagonal().std() * 1e10) != 182135562992:
    print('************** PB in DICIntegration G')
if int(m.phix.diagonal().sum() * 1e10) != 22894736842 \
or int(m.phiy.diagonal().std() * 1e10) != 160595932:
    print('************** PB in DICIntegration')

U = px.MultiscaleInit(f, g, m, cam, scales=[3,2,1])
print('SCALE  3\nIter #  9 | std(res)=5.23 gl | dU/U=8.23e-04')
print('SCALE  2\nIter #  5 | std(res)=7.43 gl | dU/U=8.25e-04')
print('SCALE  1\nIter #  4 | std(res)=6.52 gl | dU/U=2.53e-04')

U, res = px.Correlate(f, g, m, cam, U0=U)
print('Iter #  3 | std(res)=2.28 gl | dU/U=2.93e-04')

#%% Tests Postpro  = Visual Check
postpro_test(f, g, m, cam, U, res)

#%% Calibration Routines

if INTERACTIVE:
    go = input('Do You Wan to test the calibration routines? (Y/N):')
    if go == 'Y':
        ls = px.LSCalibrator(f, m)
        ls.NewCircle()
        ls.NewLine()
        ls.NewLine()
        ls.FineTuning()
        cam = ls.Calibration()
        # cam = px.Camera(np.array([-10.528098,-50.848452,6.430594,-1.574282]))

if PLOT:
    px.PlotMeshImage(f, m, cam)

#%% ONE RECTANGLE
m.e[3] = m.e[3][[0],:]
one_element_integration_test(m, cam)

#%% TRI3 mesh
m = px.ReadMeshGMSH('gmsh_t3_mm.msh')
cam = px.Camera(np.array([-10.528098, -50.848452, 6.430594, -1.574282]))
if PLOT:
    m.Plot()
m.Connectivity()
m.DICIntegration(cam)
U = px.MultiscaleInit(f, g, m, cam, scales=[3,2,1])
print('Iter #  8 | std(res)=5.40 gl | dU/U=7.95e-04')
print('Iter #  5 | std(res)=7.48 gl | dU/U=5.09e-04')
print('Iter #  4 | std(res)=6.53 gl | dU/U=2.39e-04')

U, res = px.Correlate(f, g, m, cam, U0=U)
print('Iter #  2 | std(res)=2.29 gl | dU/U=6.17e-04')

postpro_test(f, g, m, cam, U, res)

#%% TRI3 mesh

m = px.ReadMeshGMSH('gmsh_t3_mm.msh')
m.Connectivity()
m.GaussIntegration()
K = m.Stiffness(px.Hooke([11,0.33]))

m = px.ReadMeshGMSH('gmsh_t3_mm.msh')
m.Connectivity()
m.DICIntegration(cam, G=True)
K2 = m.Stiffness(px.Hooke([11,0.33]))

print(np.max(abs(K2-K))/np.max(K))

#%% ONE TRIANGLE
m.e[2]=m.e[2][[0],:]
one_element_integration_test(m, cam)

#%% 
n = np.array([[0, 0],
              [24, 0],
              [24, 8]])

ticks_x=np.linspace(0,30,31)-0.5
ticks_y=np.linspace(0,10,11)-0.5

e={2: np.array([[0,1,2]])}
m=px.Mesh(e,n)
# m.Plot()
cam = px.Camera(np.array([1., 0, 0, 0.]))
m.Connectivity()

m.GaussIntegration()
plt.figure()
m.Plot()
plt.plot(m.pgx,m.pgy,'k.')
plt.title('GaussIntegration')

m.DICIntegration(cam, G=True)
fig,axes=plt.subplots(1,1)
m.Plot()
plt.plot(m.pgx,m.pgy,'k.')
plt.title('DICIntegration (Standard)')
axes.set_xticks(ticks_x)
axes.set_yticks(ticks_y)
axes.grid()
print(m.npg)

m.DICIntegration(cam, G=True, tri_same=True)
fig,axes=plt.subplots(1,1)
m.Plot()
plt.plot(m.pgx,m.pgy,'k.')
plt.title('DICIntegration (tri_same)')
axes.set_xticks(ticks_x)
axes.set_yticks(ticks_y)
axes.grid()
print(m.npg)

m.DICIntegrationFast(cam, G=True)
fig,axes=plt.subplots(1,1)
m.Plot()
plt.plot(m.pgx,m.pgy,'k.')
plt.title('DICIntegration (Standard)')
axes.set_xticks(ticks_x)
axes.set_yticks(ticks_y)
axes.grid()
print(m.npg)

m.DICIntegrationPixel(cam)
fig,axes=plt.subplots(1,1)
m.Plot()
plt.plot(m.pgx,m.pgy,'k.')
plt.title('Pixel Integration')
axes.set_xticks(ticks_x)
axes.set_yticks(ticks_y)
axes.grid()

#%% Mix of QUA4 and Triangles

fn=os.path.join('C:\\','Users','passieux','Documents','ICA','Supervized-M2R',\
                      '20-Mohammed ElMourabit','pyxel-master','data','DIC_LSCM_Zone0',\
                          'Mesh_In718-Zone1_coarse3.inp')
m, elset = px.ReadMeshINPwithElset(fn)
if PLOT:
    m.Plot()
p = np.array([1.96737087e+00, 2.91505872e+02, -2.22102909e+02, -2.47410146e-03])
cam = px.Camera(p)
m.Connectivity()
# m.DICIntegration(cam)
m.DICIntegrationFast(cam)
m.Plot()
plt.plot(m.pgx, m.pgy, 'k.')

px.PlotMeshImage(f, m, cam)

U = px.MultiscaleInit(f, g, m, cam, scales=[3,2,1], l0=30)
l0 = 20
L = m.Tikhonov()
U, res = px.Correlate(f, g, m, cam, U0=U, L=L, l0=l0)

postpro_test(f, g, m, cam, U, res)

