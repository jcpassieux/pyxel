# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

import numpy as np
import matplotlib.pyplot as plt
from .image import Volume

class ExportPixMap:
    """ Class ExportPixMap """
    def __init__(self, f, m, cam):
        """
        Class : Export Results to a Pixel Map
        
        Parameters
        ----------
        f : PYXEL.IMAGE
            image de ref
        m : PYXEL.MESH
            mesh
        cam : PYXEL.CAMERA
            camera model

        """
        self.m = m.Copy()
        self.cam = cam
        self.m.DICIntegrationPixel(cam)
        u, v = self.cam.P(self.m.pgx, self.m.pgy)
        self.u = np.round(u).astype(int)
        self.v = np.round(v).astype(int)
        self.sizeim = f.pix.shape

    def GetUmap(self, U):
        Umap = np.zeros(self.sizeim)
        Vmap = np.zeros(self.sizeim)
        upg = self.m.phix @ U
        vpg = self.m.phiy @ U
        Umap[self.v, self.u] = upg
        Vmap[self.v, self.u] = vpg
        return Umap, Vmap

    def GetROI(self):
        Rmap = np.zeros(self.sizeim)
        roi = self.m.phix @ np.ones(self.m.ndof)
        Rmap[self.v, self.u] = (roi > 0.5).astype(int)
        return Rmap

    def GetResidual(self, f, g, U):
        Rmap = np.zeros(self.sizeim)
        imref = f.Interp(self.u, self.v)
        upg = self.m.phix @ U
        vpg = self.m.phiy @ U
        u, v = self.cam.P(self.m.pgx + upg, self.m.pgy + vpg)
        res = g.Interp(u, v)
        res -= np.mean(res)
        res = imref - np.mean(imref) - np.std(imref) / np.std(res) * res
        Rmap[self.v, self.u] = res
        return Rmap

    def PlotDispl(self, U):
        Umap, Vmap = self.GetUmap(U)
        plt.figure()
        plt.imshow(Umap)
        plt.colorbar()
        plt.title('Displacement U')

        plt.figure()
        plt.imshow(Vmap)
        plt.colorbar()
        plt.title('Displacement V')
        return Umap, Vmap

    def PlotResidual(self, f, g, U):
        R = self.GetResidual(f, g, U)
        plt.figure()
        plt.imshow(R, cmap="RdBu")
        stdr = np.std(R[self.v, self.u])
        plt.clim(-3 * stdr, 3 * stdr)
        plt.colorbar()
        plt.title('Residual Map')
        return R


class ExportVoxMap:
    """ Class ExportVoxMap """
    def __init__(self, f, m, cam):
        """
        Class : Export Results to a Voxel Map
        
        Parameters
        ----------
        f : PYXEL.IMAGE
            image de ref
        m : PYXEL.MESH
            mesh
        cam : PYXEL.CAMERA
            camera model

        """
        self.m = m.Copy()
        self.cam = cam
        self.m.DVCIntegrationTetVoxel(f, cam)
        u, v, w = self.cam.P(self.m.pgx, self.m.pgy, self.m.pgz)
        self.u = np.round(u).astype(int)
        self.v = np.round(v).astype(int)
        self.w = np.round(w).astype(int)
        self.sizeim = f.pix.shape

    def GetUmap(self, U):
        Umap = np.zeros(self.sizeim)
        Vmap = np.zeros(self.sizeim)
        Wmap = np.zeros(self.sizeim)
        upg = self.m.phix @ U
        vpg = self.m.phiy @ U
        wpg = self.m.phiz @ U
        Umap[self.u, self.v, self.w] = upg
        Vmap[self.u, self.v, self.w] = vpg
        Wmap[self.u, self.v, self.w] = wpg
        return Umap, Vmap, Wmap

    def GetROI(self):
        Rmap = np.zeros(self.sizeim)
        roi = self.m.phix @ np.ones(self.m.ndof)
        Rmap[self.v, self.u, self.w] = (roi > 0.5).astype(int)
        return Rmap

    def GetResidual(self, f, g, U):
        Rmap = np.zeros(self.sizeim)
        imref = f.Interp(self.u, self.v, self.w)
        upg = self.m.phix @ U
        vpg = self.m.phiy @ U
        wpg = self.m.phiz @ U
        u, v, w = self.cam.P(self.m.pgx + upg, self.m.pgy + vpg, self.m.pgz + wpg)
        res = g.Interp(u, v, w)
        res -= np.mean(res)
        res = imref - np.mean(imref) - np.std(imref) / np.std(res) * res
        Rmap[self.u, self.v, self.w] = res
        return Rmap

    def PlotDispl(self, U, vmax=None):
        Umap, Vmap, Wmap = self.GetUmap(U)
        h = Volume('')
        h.pix = Umap
        if vmax is None:
            vmax0 = 0.95*np.max(abs(Umap))
        else:
            vmax0 = vmax
        plt.figure()
        h.Plot(cmap='RdBu', vmin=-vmax0, vmax=vmax0)

        h.pix = Vmap
        if vmax is None:
            vmax0 = 0.95*np.max(abs(Vmap))
        else:
            vmax0 = vmax
        plt.figure()
        h.Plot(cmap='RdBu', vmin=-vmax0, vmax=vmax0)

        h.pix = Wmap
        if vmax is None:
            vmax0 = 0.95*np.max(abs(Wmap))
        else:
            vmax0 = vmax
        plt.figure()
        h.Plot(cmap='RdBu', vmin=-vmax0, vmax=vmax0)
        return Umap, Vmap, Wmap

    def PlotResidual(self, f, g, U):
        R = self.GetResidual(f, g, U)
        h = Volume('')
        h.pix = R
        vmax = 3*np.std(R)
        h.Plot(cmap='RdBu', vmin=-vmax, vmax=vmax)
        return R