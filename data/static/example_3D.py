# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:25:49 2022

@author: passieux
"""

import os
import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg


#%% Testcase with 3D solid elements

fn = os.path.join('hex8.inp')
# or
fn = os.path.join('tet4.msh')
# or
fn = os.path.join('hex20.msh')
# or
fn = os.path.join('tet10.msh')

m = px.ReadMesh(fn, 3)
m.KeepVolElems()

# m.Write('Mesh.vtu')
m.Connectivity()
C = px.Hooke([1, 0.3], 'isotropic_3D')
m.GaussIntegration()
K = m.Stiffness(C)

# Dirichlet BC at y = -0.035
repu, = np.where(m.n[:,1] < -0.03499)
repu = m.conn[repu, :]

# Dirichlet BC at y = 0.035 second dof only
repf, = np.where(m.n[:,1] > 0.03499)
repf = m.conn[repf, 1]

U = np.zeros(m.ndof)
U[repf] = 0.001
U[repu] = 0.0

rep = np.setdiff1d(np.arange(m.ndof), np.append(repu, repf))
repk = np.ix_(rep, rep)
F = -K@U

KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

m.VTKSol('Sol3D', U)




#%% Generating hex20 mesh with GMSH
import gmsh

gmsh.initialize()
gmsh.clear()
gmsh.model.add("box")
xmin = -25e-3
xmax = 25e-3
ymin = -35e-3
ymax = 35e-3
zmin = 0
zmax = 2e-3
rad = 14e-3
lc = 2e-3
gmsh.model.geo.addPoint( xmin, ymin, zmin, lc, 1)
gmsh.model.geo.addPoint( xmax, ymin, zmin, lc, 2)
gmsh.model.geo.addPoint( xmax, ymax, zmin, lc, 3)
gmsh.model.geo.addPoint( xmin, ymax, zmin, lc, 4)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2, zmin, lc, 5)
gmsh.model.geo.addPoint( (xmin+xmax)/2+rad, (ymin+ymax)/2, zmin, lc, 6)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2+rad, zmin, lc, 7)
gmsh.model.geo.addPoint( (xmin+xmax)/2-rad, (ymin+ymax)/2, zmin, lc, 8)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2-rad, zmin, lc, 9)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,3,2)
gmsh.model.geo.addLine(3,4,3)
gmsh.model.geo.addLine(4,1,4)
#gmsh.fltk.run()
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)
gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8],1)
gmsh.model.geo.addPlaneSurface([1],1)
gmsh.option.setNumber('Mesh.RecombineAll', 1)
gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
gmsh.option.setNumber('Mesh.Recombine3DLevel', 2)
gmsh.model.geo.extrude([(2, 1)], 0, 0, zmax)
gmsh.option.setNumber('Mesh.ElementOrder', 2)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("tet10.msh")
gmsh.finalize()

#%% Generating Tet4 and tet10 mesh with GMSH
import gmsh

gmsh.initialize()
gmsh.clear()
gmsh.model.add("box")
xmin = -25e-3
xmax = 25e-3
ymin = -35e-3
ymax = 35e-3
zmin = 0
zmax = 2e-3
rad = 14e-3
lc = 2e-3
gmsh.model.geo.addPoint( xmin, ymin, zmin, lc, 1)
gmsh.model.geo.addPoint( xmax, ymin, zmin, lc, 2)
gmsh.model.geo.addPoint( xmax, ymax, zmin, lc, 3)
gmsh.model.geo.addPoint( xmin, ymax, zmin, lc, 4)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2, zmin, lc, 5)
gmsh.model.geo.addPoint( (xmin+xmax)/2+rad, (ymin+ymax)/2, zmin, lc, 6)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2+rad, zmin, lc, 7)
gmsh.model.geo.addPoint( (xmin+xmax)/2-rad, (ymin+ymax)/2, zmin, lc, 8)
gmsh.model.geo.addPoint( (xmin+xmax)/2, (ymin+ymax)/2-rad, zmin, lc, 9)
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addLine(2,3,2)
gmsh.model.geo.addLine(3,4,3)
gmsh.model.geo.addLine(4,1,4)
#gmsh.fltk.run()
gmsh.model.geo.addCircleArc(6, 5, 7, 5)
gmsh.model.geo.addCircleArc(7, 5, 8, 6)
gmsh.model.geo.addCircleArc(8, 5, 9, 7)
gmsh.model.geo.addCircleArc(9, 5, 6, 8)
gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7,8],1)
gmsh.model.geo.addPlaneSurface([1],1)
gmsh.model.geo.extrude([(2, 1)], 0, 0, zmax)
gmsh.model.geo.synchronize()
#gmsh.option.setNumber('Mesh.ElementOrder', 2)
gmsh.model.mesh.generate(3)
#gmsh.write("tet4.msh")
#gmsh.write("tet10.msh")
gmsh.finalize()
