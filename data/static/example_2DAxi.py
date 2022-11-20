# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:25:49 2022

@author: passieux
"""

import numpy as np
import pyxel as px
import scipy.sparse.linalg as splalg


#%% Testcase with 2D axisymmetric

fn = 'tri3_axi.msh'
m = px.ReadMesh(fn)
m.KeepSurfElems()

# m.Write('Mesh.vtu')
m.Connectivity()
C = px.Hooke([1e6, 0.3], 'isotropic_2D_axi')
m.GaussIntegration()
K = m.StiffnessAxi(C)

# Dirichlet BC at y = -0.035
repu, = np.where(m.n[:,1] < -0.03499)
repu = m.conn[repu, 1]

# Dirichlet BC at y = 0.035 second dof only
repf, = np.where(m.n[:,0] > 0.02499)
repf = m.conn[repf, 0]

U = np.zeros(m.ndof)
F = np.zeros(m.ndof)
F[repf] = 1

rep = np.setdiff1d(np.arange(m.ndof), repu)
repk = np.ix_(rep, rep)
F -= K@U

KLU = splalg.splu(K[repk])
U[rep] = KLU.solve(F[rep])

m.VTKSol('Sol2DAxi', U)



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
#gmsh.fltk.run()
gmsh.model.geo.addCurveLoop([1,2,3,4,5,6,7],1)
gmsh.model.geo.addPlaneSurface([1],1)
#gmsh.option.setNumber('Mesh.RecombineAll', 1)
#gmsh.option.setNumber('Mesh.RecombinationAlgorithm', 1)
#gmsh.option.setNumber('Mesh.ElementOrder', 2)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("tri3_axi.msh")
gmsh.finalize()
