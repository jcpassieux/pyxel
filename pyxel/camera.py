# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""

import numpy as np

class Camera:
    def __init__(self, p):
        self.set_p(p)

    def set_p(self, p):
        """
        Set manually the camera parameters.

        Parameters
        ----------
        p : NUMPY.ARRAY
            np.array([f, tx, ty, rz])
            f: focal length (scaling)
            tx: x-translation
            ty: y-translation
            rz: z-rotation

        """
        self.f = p[0]
        self.tx = p[1]
        self.ty = p[2]
        self.rz = p[3]

    def get_p(self):
        """
        Returns the vector of parameters of the camera model

        Returns
        -------
        NUMPY.ARRAY

        """
        return np.array([self.f, self.tx, self.ty, self.rz])

    def SubSampleCopy(self, nscale):
        """
        Camera model copy with subsampling (of bining) for multiscale initialization
        
        Parameters
        ----------
        NSCALE : INT
            number of scales, such that the scaling is 2**NSCALE

        Returns
        -------
        CAM : 1D PYXEL.CAMERA
            A new camera model that maps the same mesh coord. sys. to the  
            same image but with a bining of NSCALE. 
        """
        p = self.get_p()
        p[0] /= 2 ** nscale
        return Camera(p)

    def P(self, X, Y):
        """
        Camera model projection. Maps a point of the mesh to a point
        in the image plane

        Parameters
        ----------
        X : NUMPY.ARRAY
            X coordinate in the mesh system of the points to map
        Y : NUMPY.ARRAY
            Y coordinate in the mesh system of the points to map

        Returns
        -------
        u : 1D NUMPY.ARRAY
            u coordinate of the corresponding point in the image system
        v : 1D NUMPY.ARRAY
            v coordinate of the corresponding point in the image system

        """
        u = -self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        v = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx)
        return u, v

    def Pinv(self, u, v):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (Explicit version for 2D camera model)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system

        """
        X = -np.sin(self.rz) * (-u / self.f - self.ty) + np.cos(self.rz) * (
            v / self.f - self.tx
        )
        Y = np.cos(self.rz) * (-u / self.f - self.ty) + np.sin(self.rz) * (
            v / self.f - self.tx
        )
        return X, Y

    def PinvNL(self, u, v):
        """
        Inverse of the Camera model. Maps a point in the image to a point
        in the mesh coordinate sys.
        (General version for any possibly NL camera models)

        Parameters
        ----------
        u : 1D NUMPY.ARRAY
            u coordinate in the image system of the points to map
        v : 1D NUMPY.ARRAY
            v coordinate in the image system of the points to map

        Returns
        -------
        X : 1D NUMPY.ARRAY
            X coordinate of the corresponding position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the corresponding position in the mesh system

        """
        X = np.zeros(len(u))
        Y = np.zeros(len(u))
        for ii in range(10):
            pnx, pny = self.P(X, Y)
            resx = u - pnx
            resy = v - pny
            dPxdX, dPxdY, dPydX, dPydY = self.dPdX(X, Y)
            detJ = dPxdX * dPydY - dPxdY * dPydX
            dX = dPydY / detJ * resx - dPxdY / detJ * resy
            dY = -dPydX / detJ * resx + dPxdX / detJ * resy
            X += dX
            Y += dY
            res = np.linalg.norm(dX) + np.linalg.norm(dY)
            if res < 1e-4:
                break
        return X, Y

    def dPdX(self, X, Y):
        """
        Derivative of the Camera model wrt physical position X, Y

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudx : NUMPY.ARRAY
            Derivative of coord. u wrt to X
        dudy : NUMPY.ARRAY
            Derivative of coord. u wrt to Y
        dvdx : NUMPY.ARRAY
            Derivative of coord. v wrt to X
        dvdy : NUMPY.ARRAY
            Derivative of coord. v wrt to Y

        """
        dudx = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        dudy = -self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdx = self.f * np.cos(self.rz) * np.ones(X.shape[0])
        dvdy = self.f * np.sin(self.rz) * np.ones(X.shape[0])
        return dudx, dudy, dvdx, dvdy

    def dPdp(self, X, Y):
        """
        First order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        dudp : NUMPY.ARRAY
            Derivative of coord. u wrt to p
        dvdp : NUMPY.ARRAY
            Derivative of coord. v wrt to p

        """
        dudf = -1 * (-np.sin(self.rz) * X + np.cos(self.rz) * Y + self.ty)
        dudtx = 0 * X
        dudty = 0 * X - self.f
        dudrz = self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y)
        dvdf = np.cos(self.rz) * X + np.sin(self.rz) * Y + self.tx
        dvdtx = 0 * X + self.f
        dvdty = 0 * X
        dvdrz = self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y)
        return np.c_[dudf, dudtx, dudty, dudrz], np.c_[dvdf, dvdtx, dvdty, dvdrz]

    def d2Pdp2(self, X, Y):
        """
        Second order derivative of the Camera model wrt camera parameters p

        Parameters
        ----------
        X : 1D NUMPY.ARRAY
            X coordinate of the current position in the mesh system
        Y : 1D NUMPY.ARRAY
            Y coordinate of the current position in the mesh system

        Returns
        -------
        d2udp2 : NUMPY.ARRAY
            second order derivative of coord. u wrt to p
        d2vdp2 : NUMPY.ARRAY
            second order derivative of coord. v wrt to p

        """
        d2udf2 = 0 * X
        d2udtx2 = 0 * X
        d2udty2 = 0 * X
        d2udrz2 = self.f * (-np.sin(self.rz) * X + np.cos(self.rz) * Y)
        d2udftx = 0 * X
        d2udfty = 0 * X - 1
        d2udfrz = np.cos(self.rz) * X + np.sin(self.rz) * Y
        d2udtxty = 0 * X
        d2udtxrz = 0 * X
        d2udtyrz = 0 * X
        d2vdf2 = 0 * X
        d2vdtx2 = 0 * X
        d2vdty2 = 0 * X
        d2vdrz2 = -self.f * (np.cos(self.rz) * X + np.sin(self.rz) * Y)
        d2vdftx = 0 * X + 1
        d2vdfty = 0 * X
        d2vdfrz = -np.sin(self.rz) * X + np.cos(self.rz) * Y
        d2vdtxty = 0 * X
        d2vdtxrz = 0 * X
        d2vdtyrz = 0 * X
        d2udp2 = np.c_[d2udf2, d2udtx2, d2udty2, d2udrz2, d2udftx, 
                       d2udfty, d2udfrz, d2udtxty, d2udtxrz, d2udtyrz]
        d2vdp2 = np.c_[d2vdf2, d2vdtx2, d2vdty2, d2vdrz2, d2vdftx,
                       d2vdfty, d2vdfrz, d2vdtxty, d2vdtxrz, d2vdtyrz]
        return d2udp2, d2vdp2

    def ImageFiles(self, fname, imnums):
        self.fname = fname
        self.imnums = imnums
