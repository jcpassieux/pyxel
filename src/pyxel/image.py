# -*- coding: utf-8 -*-
"""
Finite Element Digital Image Correlation method 

@author: JC Passieux, INSA Toulouse, 2021

pyxel

PYthon library for eXperimental mechanics using Finite ELements

"""
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import PIL.Image as image
from .utils import PlotMeshImage

class Image:
    def Load(self):
        """Load image data"""
        if self.fname.split(".")[-1] == "npy":
            self.pix = np.load(self.fname)
        else:
            self.pix = np.asarray(image.open(self.fname)).astype(float)
            # self.pix = image.imread(self.fname).astype(float)
        if len(self.pix.shape) == 3:
            self.ToGray()
        return self

    def Load_cv2(self):
        """Load image data using OpenCV"""
        import cv2 as cv
        self.pix = cv.imread(self.fname).astype(float)
        if len(self.pix.shape) == 3:
            self.ToGray()
        return self

    def Copy(self):
        """Image Copy"""
        newimg = Image("Copy")
        newimg.pix = self.pix.copy()
        return newimg

    def Save(self, fname):
        """Image Save"""
        PILimg = image.fromarray(self.pix.astype("uint8"))
        PILimg.save(fname)
        # image.imsave(fname,self.pix.astype('uint8'),vmin=0,vmax=255,format='tif')

    def __init__(self, fname):
        """Contructor"""
        self.fname = fname

    def BuildInterp(self):
        """build bivariate Spline interp"""
        x = np.arange(0, self.pix.shape[0])
        y = np.arange(0, self.pix.shape[1])
        self.tck = spi.RectBivariateSpline(x, y, self.pix)

    def Interp(self, x, y):
        """evaluate interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y)

    def InterpGrad(self, x, y):
        """evaluate gradient of the interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y, 1, 0), self.tck.ev(x, y, 0, 1)

    def InterpHess(self, x, y):
        """evaluate Hessian of the interpolator at non-integer pixel position x, y"""
        return self.tck.ev(x, y, 2, 0), self.tck.ev(x, y, 0, 2), self.tck.ev(x, y, 1, 1)

    def Plot(self):
        """Plot Image"""
        plt.imshow(self.pix, cmap="gray", interpolation="none", origin="upper")
        # plt.axis('off')
        # plt.colorbar()

    def Dynamic(self):
        """Compute image dynamic"""
        g = self.pix.ravel()
        return max(g) - min(g)

    def GaussianFilter(self, sigma=0.7):
        """Performs a Gaussian filter on image data. 

        Parameters
        ----------
        sigma : float
            variance of the Gauss filter."""
        from scipy.ndimage import gaussian_filter

        self.pix = gaussian_filter(self.pix, sigma)

    def PlotHistogram(self):
        """Plot Histogram of graylevels"""
        plt.hist(self.pix.ravel(), bins=125, range=(0.0, 255), fc="k", ec="k")
        plt.show()

    def SubSample(self, n):
        """Image copy with subsampling for multiscale initialization"""
        scale = 2 ** n
        sizeim1 = np.array([self.pix.shape[0] // scale, self.pix.shape[1] // scale])
        nn = scale * sizeim1
        im0 = np.mean(
            self.pix[0 : nn[0], 0 : nn[1]].T.reshape(np.prod(nn) // scale, scale),
            axis=1,
        )
        nn[0] = nn[0] // scale
        im0 = np.mean(
            im0.reshape(nn[1], nn[0]).T.reshape(np.prod(nn) // scale, scale), axis=1
        )
        nn[1] = nn[1] // scale
        self.pix = im0.reshape(nn)

    def ToGray(self, type="lum"):
        """Convert RVG to Grayscale :

        Parameters
        ----------
        type : string
            lig : lightness
            lum : luminosity (DEFAULT)
            avg : average"""
        if type == "lum":
            self.pix = (
                0.21 * self.pix[:, :, 0]
                + 0.72 * self.pix[:, :, 1]
                + 0.07 * self.pix[:, :, 2]
            )
        elif type == "lig":
            self.pix = 0.5 * np.maximum(
                np.maximum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            ) + 0.5 * np.minimum(
                np.minimum(self.pix[:, :, 0], self.pix[:, :, 1]), self.pix[:, :, 2]
            )
        else:
            self.pix = np.mean(self.pix, axis=2)

    def SelectPoints(self, n=-1, title=None):
        """Select a point in the image. 
        
        Parameters
        ----------
        n : int
            number of expected points
        title : string (OPTIONNAL)
            modify the title of the figure when clic is required.
            
        """
        plt.figure()
        self.Plot()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if title is None:
            if n < 0:
                plt.title("Select some points... and press enter")
            else:
                plt.title("Select " + str(n) + " points... and press enter")
        else:
            plt.title(title)
        pts1 = np.array(plt.ginput(n, timeout=0))
        plt.close()
        return pts1

    def SelectROI(self, m=None, cam=None):
        """Select a Region of Interest within the image. 
        
        Parameters
        ----------
        m : pyxel.Mesh object (OPTIONNAL)
        cam : pyxel.Camera object (OPTIONNAL)
            To superimpose the mesh in the image
        
        The result of the ROI is displayed in the python command. 
        It can be copy-pasted.
        """
        from matplotlib.widgets import RectangleSelector

        fig, ax = plt.subplots()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        if m is not None:
            PlotMeshImage(self, m, cam, newfig=False)
        else:
            self.Plot()

        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print(
                "roi = np.array([[%4d, %4d], [%4d, %4d]])"
                % (int(x1), int(y1), int(x2), int(y2))
            )

        rs = RectangleSelector(
            ax,
            line_select_callback,
            drawtype="box",
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        plt.show()
        return rs
