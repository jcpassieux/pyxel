# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:25:49 2022

@author: passieux
"""

import os
import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg
import gmsh


# %% Testcase with 2D plane stress elasticity
test = 'tri3'
# or
test = 'tri6'
# or
test = 'qua4'
# or
test = 'qua8'

fn = os.path.join(test + '.msh')
m = px.ReadMesh(fn)
m.KeepSurfElems()

# m.Write('Mesh.vtu')
m.Connectivity()
# C = px.Hooke([1, 0.3], 'isotropic_2D_pe') # Plane strain
C = px.Hooke([1, 0.3], 'isotropic_2D_ps')  # Plane stress
m.GaussIntegration()
K = m.Stiffness(C)

# Dirichlet BC at y = -0.035
repb = m.SelectEndLine('bottom', 1e-5)
rept = m.SelectEndLine('top', 1e-5)

BC = [[repb, [[0, 0], [1, 0]]],   # setting all dof to zero on bottom line
      [rept, [[1, 0.001], ]]]       # setting y-dof of top line to 0.1

U, RF = m.SolveElastic(K, BC, LOAD=[])

# %% Post-Processing > Matplotlib

m.Plot(alpha=0.1)
m.Plot(U, 10)
m.PlotContourDispl(U, s=10, plotmesh=False)
m.PlotContourStrain(U, s=10, clim=[-0.02, 0.02], cmap='RdBu', plotmesh=False)
m.PlotContourStress(U, C, clim=[-0.02, 0.02], cmap='RdBu', plotmesh=False)

# %% Post-processing > Paraview

m.Write('mesh.vtu')
m.VTKSol('Sol2D_'+test, U)

# %% Generating 2D meshes with GMSH

# test = 'tri3'
# # or
# test = 'tri6'
# # or
# test = 'qua4'
# # or
# test = 'qua8'

# gmsh.initialize()
# gmsh.clear()
# gmsh.model.add("P")
# xmin = -25e-3
# xmax = 25e-3
# ymin = -35e-3
# ymax = 35e-3
# zmin = 0
# zmax = 2e-3
# rad = 14e-3
# lc = 2e-3
# gmsh.model.geo.addPoint(xmin, ymin, zmin, lc, 1)
# gmsh.model.geo.addPoint(xmax, ymin, zmin, lc, 2)
# gmsh.model.geo.addPoint(xmax, ymax, zmin, lc, 3)
# gmsh.model.geo.addPoint(xmin, ymax, zmin, lc, 4)
# gmsh.model.geo.addPoint((xmin+xmax)/2, (ymin+ymax)/2, zmin, lc, 5)
# gmsh.model.geo.addPoint((xmin+xmax)/2+rad, (ymin+ymax)/2, zmin, lc, 6)
# gmsh.model.geo.addPoint((xmin+xmax)/2, (ymin+ymax)/2+rad, zmin, lc, 7)
# gmsh.model.geo.addPoint((xmin+xmax)/2-rad, (ymin+ymax)/2, zmin, lc, 8)
# gmsh.model.geo.addPoint((xmin+xmax)/2, (ymin+ymax)/2-rad, zmin, lc, 9)
# gmsh.model.geo.addLine(1, 2, 1)
# gmsh.model.geo.addLine(2, 3, 2)
# gmsh.model.geo.addLine(3, 4, 3)
# gmsh.model.geo.addLine(4, 1, 4)
# gmsh.model.geo.addCircleArc(6, 5, 7, 5)
# gmsh.model.geo.addCircleArc(7, 5, 8, 6)
# gmsh.model.geo.addCircleArc(8, 5, 9, 7)
# gmsh.model.geo.addCircleArc(9, 5, 6, 8)
# gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8], 1)
# gmsh.model.geo.addPlaneSurface([1], 1)
# if 'qua' in test:
#     gmsh.option.setNumber('Mesh.RecombineAll', 1)
#     gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
# if test[-1] in ['6', '8']:
#     gmsh.option.setNumber('Mesh.ElementOrder', 2)
# gmsh.model.geo.synchronize()
# gmsh.model.mesh.generate(2)
# gmsh.write(test+'.msh')
# # gmsh.fltk.run()
# gmsh.finalize()
