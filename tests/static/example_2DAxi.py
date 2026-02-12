# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:25:49 2022

@author: passieux
"""

import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg
import matplotlib.pyplot as plt

#%% Testcase with 2D axisymmetric

fn = 'tri3_axi.msh'
m = px.ReadMesh(fn)
m.KeepSurfElems()

# m.Write('Mesh.vtu')
m.Connectivity()
C = px.Hooke([1e6, 0.3], 'isotropic_2D_axi')
m.GaussIntegration()
K = m.StiffnessAxi(C)

# Dirichlet BC at the bottom
repu = m.SelectEndLine('bottom')
BC = [[repu, [[1, 0], ]], ]

# Neumann BC on the right end
fx = 3000.0  # surface traction (N/m²).
fy = 0.0
right_box = px.PointCloud2Box(m.n[m.SelectEndLine('right', plot=True)])
mb = m.BuildBoundaryMesh()
mb.RemoveElemsOutsideRoi(right_box)
m.Plot()
mb.Plot(edgecolor='r')
mb.GaussIntegration()
F = mb.wdetJ @ mb.phix * fx + mb.wdetJ @ mb.phiy * fy

U, RF = m.SolveElastic(K, BC, F)

m.Plot(U, 1)
m.PlotContourDispl(U)

# plots only the in-plane components of stress.
m.PlotContourStress(U, C)

m.VTKSol('Sol2DAxi', U)

# %%

EN, ES = m.StrainAtGP(U, axisym=True)
SN, SS = px.Strain2Stress(C, EN, ES)

plt.figure()
m.Plot()
plt.scatter(m.pgx, m.pgy, c=SN[:, 2])
plt.colorbar()
plt.axis('equal')

#%% generating axisymmetric mesh
import gmsh

gmsh.initialize()
gmsh.clear()
gmsh.model.add("P")
xmin = 0
xmax = 25e-3
ymin = -35e-3
ymax = 35e-3
zmin = 0
rad = 14e-3
lc = 2e-3
gmsh.model.geo.addPoint( xmin, ymin, zmin, lc, 1)
gmsh.model.geo.addPoint( xmax, ymin, zmin, lc, 2)
gmsh.model.geo.addPoint( xmax, ymax, zmin, lc, 3)
gmsh.model.geo.addPoint( xmin, ymax, zmin, lc, 4)
gmsh.model.geo.addPoint( xmin, (ymin+ymax)/2, zmin, lc, 5)
gmsh.model.geo.addPoint( xmin+rad, (ymin+ymax)/2, zmin, lc, 6)
gmsh.model.geo.addPoint( xmin, (ymin+ymax)/2+rad, zmin, lc, 7)
gmsh.model.geo.addPoint( xmin, (ymin+ymax)/2-rad, zmin, lc, 8)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,3,2)
gmsh.model.geo.addLine(3,4,3)
gmsh.model.geo.addLine(4,7,4)
gmsh.model.geo.addCircleArc(7, 5, 6, 5)
gmsh.model.geo.addCircleArc(6, 5, 8, 6)
gmsh.model.geo.addLine(8,1,7)
gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7],1)
gmsh.model.geo.addPlaneSurface([1],1)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("tri3_axi.msh")
gmsh.finalize()
