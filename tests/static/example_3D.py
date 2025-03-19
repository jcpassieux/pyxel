# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:25:49 2022

@author: passieux
"""

import os
import numpy as np
import pyxel as px

# %% Testcase with 3D solid elements

eltype = 'hex8'   # linear hexaedra
# or
eltype = 'hex20'  # quadratic hexaedra
# or
eltype = 'tet4'   # linear tetraedra
# or
eltype = 'tet10'  # quadratic tetraedra

fn = os.path.join(eltype+'.msh')

m = px.ReadMesh(fn, 3)
m.KeepVolElems()
# m.KeepSurfElems()
m.Plot()

# m.Write('Mesh.vtu')
m.Connectivity()
C = px.Hooke([1, 0.3], 'isotropic_3D')
m.GaussIntegration()
K = m.Stiffness(C)

# Dirichlet BC at y = -0.035
repu = m.SelectEndLine('bottom')
BC = [[repu, [[0, 0], [1, 0], [2, 0]]]]

# Dirichlet BC at y = 0.035 second dof only
repf = m.SelectEndLine('top')
BC += [[repf, [[2, -0.001]]]]

U, RF = m.SolveElastic(K, BC, [])

m.VTKSol('Sol3D_'+eltype, U)

m.Plot(U, 10)


# %% Generating hex8 and hex20 (second order incomplete) mesh with GMSH
# import gmsh
# eltype = 'hex8'   # linear hexaedra
# # or
# eltype = 'hex20'  # quadratic hexaedra

# gmsh.initialize()
# gmsh.clear()
# gmsh.model.add("box")
# xmin = -25e-3
# xmax = 25e-3
# ymin = -35e-3
# ymax = 35e-3
# zmin = 0
# zmax = 2e-3
# rad = 14e-3
# lc = 2e-3
# gmsh.model.geo.addPoint( xmin, ymin, zmin, lc, 1)
# gmsh.model.geo.addPoint( xmax, ymin, zmin, lc, 2)
# gmsh.model.geo.addPoint( xmax, ymax, zmin, lc, 3)
# gmsh.model.geo.addPoint( xmin, ymax, zmin, lc, 4)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2, zmin, lc, 5)
# gmsh.model.geo.addPoint( (xmin+xmax)/2+rad, (ymin+ymax)/2, zmin, lc, 6)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2+rad, zmin, lc, 7)
# gmsh.model.geo.addPoint( (xmin+xmax)/2-rad, (ymin+ymax)/2, zmin, lc, 8)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2-rad, zmin, lc, 9)
# gmsh.model.geo.addLine(1,2,1)
# gmsh.model.geo.addLine(2,3,2)
# gmsh.model.geo.addLine(3,4,3)
# gmsh.model.geo.addLine(4,1,4)
# #gmsh.fltk.run()
# gmsh.model.geo.addCircleArc(6, 5, 7, 5)
# gmsh.model.geo.addCircleArc(7, 5, 8, 6)
# gmsh.model.geo.addCircleArc(8, 5, 9, 7)
# gmsh.model.geo.addCircleArc(9, 5, 6, 8)
# gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8],1)
# gmsh.model.geo.addPlaneSurface([1],1)
# gmsh.option.setNumber('Mesh.RecombineAll', 1)
# gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
# gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
# gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
# gmsh.model.geo.extrude([(2, 1)], 0, 0, zmax-zmin, [2,], recombine=True)
# if eltype == 'hex20':
#     gmsh.option.setNumber('Mesh.ElementOrder', 2)
# gmsh.model.geo.synchronize()
# gmsh.model.mesh.generate(3)

# #gmsh.model.mesh.getElementsByType(14)
# gmsh.write(eltype+".msh")
# gmsh.finalize()

# %% Generating Tet4 and tet10 mesh with GMSH
# import gmsh

# eltype = 'tet4'   # linear tetraedra
# # or
# eltype = 'tet10'  # quadratic tetraedra

# gmsh.initialize()
# gmsh.clear()
# gmsh.model.add("box")
# xmin = -25e-3
# xmax = 25e-3
# ymin = -35e-3
# ymax = 35e-3
# zmin = 0
# zmax = 2e-3
# rad = 14e-3
# lc = 2e-3
# gmsh.model.geo.addPoint( xmin, ymin, zmin, lc, 1)
# gmsh.model.geo.addPoint( xmax, ymin, zmin, lc, 2)
# gmsh.model.geo.addPoint( xmax, ymax, zmin, lc, 3)
# gmsh.model.geo.addPoint( xmin, ymax, zmin, lc, 4)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2, zmin, lc, 5)
# gmsh.model.geo.addPoint( (xmin+xmax)/2+rad, (ymin+ymax)/2, zmin, lc, 6)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2+rad, zmin, lc, 7)
# gmsh.model.geo.addPoint( (xmin+xmax)/2-rad, (ymin+ymax)/2, zmin, lc, 8)
# gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2-rad, zmin, lc, 9)
# gmsh.model.geo.addLine(1,2,1)
# gmsh.model.geo.addLine(2,3,2)
# gmsh.model.geo.addLine(3,4,3)
# gmsh.model.geo.addLine(4,1,4)
# #gmsh.fltk.run()
# gmsh.model.geo.addCircleArc(6, 5, 7, 5)
# gmsh.model.geo.addCircleArc(7, 5, 8, 6)
# gmsh.model.geo.addCircleArc(8, 5, 9, 7)
# gmsh.model.geo.addCircleArc(9, 5, 6, 8)
# gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8],1)
# gmsh.model.geo.addPlaneSurface([1],1)
# gmsh.model.geo.extrude([(2, 1)], 0, 0, zmax)
# gmsh.model.geo.synchronize()
# if eltype == 'tet10':
#     gmsh.option.setNumber('Mesh.ElementOrder', 2)
# gmsh.model.mesh.generate(3)
# gmsh.write(eltype+".msh")
# gmsh.finalize()

