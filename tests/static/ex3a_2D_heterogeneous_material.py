# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:41:15 2024

Example 1: minimal testcase with two elements
> making elem sets by hand

# o  ----------------- ->
# o |        |        |->
# o |  hard  |  soft  |->
# o |        |        |->
# ///----------------- ->

@author: passieux

"""

import numpy as np
import pyxel as px

box = np.array([[0, 0], [2, 1]])
m = px.StructuredMeshQ4(box, 1)
m.Plot()

# define cell_sets as a dict [sets] of dict [elemtype]
m.cell_sets = {'hard': {3: np.array([0])},
               'soft': {3: np.array([1])}}

m.Write('test.vtu')

m.Connectivity()
m.GaussIntegration()

C = dict()
C['soft'] = px.Hooke([1, 0.3], 'isotropic_2D_ps')
C['hard'] = px.Hooke([100, 0.3], 'isotropic_2D_ps')
hooke = m.AssignMaterial2GaussPoint(C)

K = m.Stiffness(hooke)

nodes_left = [0, 1]
nodes_left_bottom = [0, ]
BC = [[nodes_left, [[0, 0], ]],         # blocking x-dof for left nodes
      [nodes_left_bottom, [[1, 0],]]]   # blocking y-dof for node 0

nodes_right = [4, 5]
LOAD = [[nodes_right, [[0, 1], ]]]      # set unit force on x-dof

u, r = m.SolveElastic(K, BC, LOAD)

# %% Post process

m.Plot(u, alpha=0.2)
m.Plot(u, 1)

m.PlotContourStrain(u, cmap='RdBu')

m.PlotContourDispl(u, s=1)

m.PlotContourStress(u, hooke)
