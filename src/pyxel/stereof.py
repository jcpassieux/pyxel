# -*- coding: utf-8 -*-
"""
Optical Flow Stereo Digital Image Correlation
(disclamer: no finite elements here)

@author: JC Passieux, INSA Toulouse, 2026

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""
import numpy as np
import pyxel as px
import cv2
import meshio
import os
from scipy.ndimage import map_coordinates

class StereOF:
    def __init__(self, rig, imfiles, mask):
        """ mask defining the ZOI (1 in, 0 out)"""
        self.rig = rig
        # x0: positions of integ. points in reference cam image at reference time
        row_mask, col_mask = np.where(mask.pix)
        self.x_c0_t0 = np.c_[col_mask, row_mask].astype('float32')
        self.imfiles = imfiles
        self.windowsize = 13
        self.finestscale = 0
        self.stepsize = 1
        self.meth = 'fine'

    def _OF(self, imf, img):
        if self.meth == 'MEDIUM':
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        elif self.meth == 'FAST':
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        elif self.meth == 'ULTRAFAST':
            flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        else:
            # MANUAL
            flow = cv2.DISOpticalFlow_create()
            flow.setVariationalRefinementAlpha(20.0)		# Laplacian of displacment
            flow.setVariationalRefinementGamma(10.0)		# Gradient of image consistency
            flow.setVariationalRefinementDelta(5.0) 	    # Optical flow
            flow.setVariationalRefinementIterations(5)	    # Number of iterations
            flow.setFinestScale(self.finestscale)
            flow.setPatchSize(self.windowsize)
            flow.setPatchStride(self.stepsize)
        UV = flow.calc(imf.pix.astype('uint8'), img.pix.astype('uint8'), None)
        flow_col = UV[::, ::, 0]
        flow_row = UV[::, ::, 1]
        return flow_col, flow_row

    def ShapeMeasurement(self):
        """
        Shape measurement from images of all cams at reference time
        u for rows
        v for cols
        c for camera
        t for time
        """
        # Finding Stereo corresponding points
        I_c0_t0 = px.Image(self.imfiles[0] % 0).Load()
        I_c1_t0 = px.Image(self.imfiles[1] % 0).Load()
        flow_col, flow_row = self._OF(I_c0_t0, I_c1_t0)
        v_c0_t0 = self.x_c0_t0[:, 0].astype('int')
        u_c0_t0 = self.x_c0_t0[:, 1].astype('int')
        u_c1_t0 = u_c0_t0 + flow_row[u_c0_t0, v_c0_t0]  # inverse indexing
        v_c1_t0 = v_c0_t0 + flow_col[u_c0_t0, v_c0_t0]
        x_c1_t0 = np.c_[v_c1_t0, u_c1_t0].astype('float32')
        # Triangulation
        self.X_t0 = self.rig.Triangulation(self.x_c0_t0, x_c1_t0)
        # CHECK REPROJECTION OK
        # x_c0, x_c1 = self.rig.Projection(self.X_t0)
        # x_c0 = x_c0[:, 0, :]
        # x_c1 = x_c1[:, 0, :]
        # print('repro error 0: mean = %2.3f and std = %2.3f' % (np.mean(x_c0 - self.x_c0_t0), np.std(x_c0 - self.x_c0_t0)))
        # print('repro error 1: mean = %2.3f and std = %2.3f' % (np.mean(x_c1 - x_c1_t0), np.std(x_c1 - x_c1_t0)))
        return self.X_t0
    
    def DisplacementMeasurement(self, time=1):
        I_c0_t0 = px.Image(self.imfiles[0] % 0).Load()
        I_c0_t1 = px.Image(self.imfiles[0] % time).Load()
        I_c1_t1 = px.Image(self.imfiles[1] % time).Load()
        # Temporal trackings
        v_c0_t0 = self.x_c0_t0[:, 0]
        u_c0_t0 = self.x_c0_t0[:, 1]
        rc = np.array([u_c0_t0, v_c0_t0])
        def track(I_a, I_b):
            flow_col, flow_row = self._OF(I_a, I_b)
            u = u_c0_t0 + map_coordinates(flow_row, rc, order=1, mode='nearest')
            v = v_c0_t0 + map_coordinates(flow_col, rc, order=1, mode='nearest')
            return np.c_[v, u].astype('float32')   # (x, y) pour OpenCV
        x_c0_t1 = track(I_c0_t0, I_c0_t1)          # cam0 t0 -> cam0 t1
        x_c1_t1 = track(I_c0_t0, I_c1_t1)          # cam0 t0 -> cam1 t1
        # Triangulation
        X_t1 = self.rig.Triangulation(x_c0_t1, x_c1_t1)
        U = X_t1 - self.X_t0
        return U
    
    def GetImgAtPoints(self, X=None, time=0):
        if X is None:
            X = self.X_t0
        x_c0, x_c1 = self.rig.Projection(X)
        x_c0 = x_c0[:, 0, :].T
        x_c1 = x_c1[:, 0, :].T
        I_c0 = px.Image(self.imfiles[0] % time).Load()
        I_c1 = px.Image(self.imfiles[1] % time).Load()
        im0 = map_coordinates(I_c0.pix, np.flip(x_c0, axis=0), order=1, mode='nearest')
        im1 = map_coordinates(I_c1.pix, np.flip(x_c1, axis=0), order=1, mode='nearest')
        return im0, im1

    def VTKfile(self, filename, fields):
        nnode = self.X_t0.shape[0]
        cells = {'vertex': np.arange(nnode)[np.newaxis].T}
        points = self.X_t0
        mesh = meshio.Mesh(points, cells)
        mesh.cell_data = {}
        mesh.point_data = {}
        mesh.point_data = fields
        dir0, filename = os.path.split(filename)
        if not os.path.isdir(os.path.join("vtk", dir0)):
            os.makedirs(os.path.join("vtk", dir0))
        mesh.write(os.path.join("vtk", dir0, filename)+'.vtu')
        print("Meshfile " + os.path.join("vtk", dir0, filename)
                + '.vtu' + " written.")