
import numpy as np
import matplotlib.pyplot as plt
from .image import Image
from .camera import Camera

#%% LevelSet Calibration tools

def LSfromLine(f, pts1):
    """Compute level set of a line from a point cloud"""
    b = pts1.T.dot(np.ones_like(pts1[:, 1]))
    A = pts1.T.dot(pts1)
    res = np.linalg.solve(A, b)
    ui = np.arange(0, f.pix.shape[0])
    vi = np.arange(0, f.pix.shape[1])
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvlset = (Xi * res[0] + Yi * res[1] - 1) / np.linalg.norm(res)
    lvl = Image("lvl")
    lvl.pix = abs(lvlset)
    return lvl

def LSfromPoint(f, pts1):
    """Compute level set from one single point"""
    pts1 = pts1.ravel()
    ui = np.arange(f.pix.shape[0]) - pts1[0]
    vi = np.arange(f.pix.shape[1]) - pts1[1]
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvl = Image("lvl")
    lvl.pix = np.sqrt(Xi ** 2 + Yi ** 2)
    return lvl

def LSfromCircle(f, pts1):
    """Compute level set of a circle from a point cloud"""
    meanu = np.mean(pts1, axis=0)
    pts = pts1 - meanu
    pts2 = pts ** 2
    A = pts.T.dot(pts)
    b = 0.5 * np.sum(pts.T.dot(pts2), axis=1)
    cpos = np.linalg.solve(A, b)
    R = np.sqrt(np.linalg.norm(cpos) ** 2 + np.sum(pts2) / pts.shape[0])
    cpos += meanu
    ui = np.arange(0, f.pix.shape[0])
    vi = np.arange(0, f.pix.shape[1])
    [Yi, Xi] = np.meshgrid(vi, ui)
    lvlset = abs(np.sqrt((Xi - cpos[0]) ** 2 + (Yi - cpos[1]) ** 2) - R)
    lvl = Image("lvl")
    lvl.pix = abs(lvlset)
    return lvl

class LSCalibrator:
    """Calibration of a front parallel setting 2D-DIC"""
    def __init__(self, f, m, cam=None):
        self.f = f
        self.m = m
        self.ptsi = dict()
        self.ptsm = dict()
        self.feat = dict()
        self.nfeat = 0
        self.lvl = dict()
        if cam is None:
            self.cam = Camera(m.dim)
        else:
            self.cam = cam

    def Init3Pts(self, ptsm=None, ptsM=None):
        """Initialization of the calibration using 3 points.

        Parameters
        ----------
        ptsm : Numpy array
            points coordinates in the images (DEFAULT = defined by clic)
        ptsM : Numpy array
            points coordinates in the mesh (DEFAULT = defined by clic)
            
        """
        if ptsm is None:
            print(" ************************************************* ")
            print(" *  SELECT 3 characteristic points in the image  * ")
            print(" ************************************************* ")
            ptsm = self.f.SelectPoints(3)
        if ptsM is None:
            print(" ************************************************* ")
            print(" * SELECT the 3 corresponding points on the mesh * ")
            print(" ************************************************* ")
            ptsM = self.m.SelectPoints(3)

        cm = np.mean(ptsm, axis=0)
        cM = np.mean(ptsM, axis=0)
        dm = np.linalg.norm(ptsm - cm, axis=1)
        dM = np.linalg.norm(ptsM - cM, axis=1)
        scale = np.mean(dm / dM)
        dmax = np.argmax(dm)
        vm = ptsm[dmax] - cm
        vM = ptsM[dmax] - cM
        vm /= np.linalg.norm(vm)
        vM /= np.linalg.norm(vM)
        angl = np.arccos(vM @ vm)
        self.cam.T[2, 0] = np.mean(dM/dm) * self.cam.K[0,0]
        self.cam.R[2, 0] = angl        
        p = self.cam.get_p()
        for i in range(40):
            up, vp = self.cam.P(ptsM[:, 0], ptsM[:, 1])
            dPudp, dPvdp = self.cam.dPdp(ptsM[:, 0], ptsM[:, 1])
            A = np.vstack((dPudp, dPvdp))
            M = A.T @ A
            b = A.T @ (ptsm.T.ravel() - np.append(up, vp))
            dp = np.linalg.solve(M, b)
            p += 0.8 * dp
            self.cam.set_p(p)
            err = np.linalg.norm(dp) / np.linalg.norm(p)
            res = np.linalg.norm(ptsm.T.ravel() - np.append(up, vp)) / \
                np.linalg.norm(ptsm.T.ravel())
            print("Iter # %2d | disc=%2.2f %% | dU/U=%1.2e" %\
                  (i + 1, res*100, err))
            if err < 1e-5:
                break

    def Plot(self):
        """Plot the level sets of each feature"""
        for i in self.feat.keys():
            plt.figure()
            self.f.Plot()
            plt.contour(self.lvl[i].pix, np.array([0.4]), colors=["y"])
            plt.figure()
            plt.contourf(self.lvl[i].pix, 16, origin="image")
            plt.colorbar()
            plt.contour(self.lvl[i].pix, np.array([0.4]), colors=["y"], origin="image")
            plt.axis("image")

    def NewCircle(self):
        print(" ******************************* ")
        print(" *        SELECT Circle        * ")
        self.ptsi[self.nfeat] = self.f.SelectPoints(
            -1, title="Select n points of a circle... and press enter"
        )  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectCircle()
        self.feat[self.nfeat] = "circle"
        self.nfeat += 1
        print(" ******************************* ")

    def NewLine(self):
        print(" ******************************* ")
        print(" *        SELECT Line          *")
        self.ptsi[self.nfeat] = self.f.SelectPoints(
            -1, title="Select n points of a straight line... and press enter"
        )  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectLine()
        self.feat[self.nfeat] = "line"
        self.nfeat += 1
        print(" ******************************* ")

    def NewPoint(self):
        print(" ******************************* ")
        print(" *        SELECT Point         * ")
        self.ptsi[self.nfeat] = self.f.SelectPoints(1)  # [:,[1,0]]
        self.ptsm[self.nfeat] = self.m.SelectNodes(1)
        self.feat[self.nfeat] = "point"
        self.nfeat += 1
        print(" ******************************* ")

    def DisableFeature(self, i):
        """Disable one of the features. Used to redefine an inappropriate mesh selection.

        Parameters
        ----------
        i : int
            the feature number
            
        """
        if i in self.feat.keys():
            del self.lvl[i]
            del self.ptsi[i]
            del self.ptsm[i]
            del self.feat[i]

    def FineTuning(self, im=None):
        """Redefine and refine the points selected in the images.

        Parameters
        ----------
        im : int (OPTIONNAL)
            the feature number is only one feature has to be redefined. Default = all
            
        """

        # Arg: f pyxel image or Array of pyxel images
        if im is None:
            rg = self.ptsi.keys()
        else:
            rg = np.array([im])
        for i in rg:  # loop on features
            for j in range(len(self.ptsi[i][:, 1])):  # loop on points
                x = int(self.ptsi[i][j, 0])
                y = int(self.ptsi[i][j, 1])
                umin = max(0, x - 50)
                vmin = max(0, y - 50)
                umax = min(self.f.pix.shape[1] - 1, x + 50)
                vmax = min(self.f.pix.shape[0] - 1, y + 50)
                fsub = self.f.pix[umin:umax, vmin:vmax]
                plt.imshow(fsub, cmap="gray", interpolation="none")
                plt.plot(x - umin, y - vmin, "y+")
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                self.ptsi[i][j, :] = np.array(plt.ginput(1))[0, ::-1]
                self.ptsi[i][j, :] += np.array([umin, vmin])
                plt.close()

    def Calibration(self, cam=None):
        """Performs the calibration provided that sufficient features have been 
        selected using NewPoint(), NewLine() or NewCircle().
            
        Returns
        -------
        pyxel Camera object
            The calibrated camera model    
        """
        # Compute Levelsets
        for i in self.feat.keys():
            if "circle" in self.feat[i]:
                self.lvl[i] = LSfromCircle(self.f, self.ptsi[i])
            elif "line" in self.feat[i]:
                self.lvl[i] = LSfromLine(self.f, self.ptsi[i])
            elif "point" in self.feat[i]:
                self.lvl[i] = LSfromPoint(self.f, self.ptsi[i])

        # Calibration
        xp = dict()
        yp = dict()
        for i in self.feat.keys():
            self.lvl[i].BuildInterp()
            xp[i] = self.m.n[self.ptsm[i], 0]
            yp[i] = self.m.n[self.ptsm[i], 1]
        # if self.cam is None:
        if len(self.feat) > 2:
            ptsm = np.empty((0, 2))
            ptsM = np.empty((0, 2))
            for i in self.feat.keys():
                ptsm = np.vstack((ptsm, np.mean(self.ptsi[i], axis=0)))
                ptsM = np.vstack((ptsM, np.mean(self.m.n[self.ptsm[i]], axis=0)))
            self.Init3Pts(ptsm, ptsM)
        else:
            self.Init3Pts()
        p = self.cam.get_p()
        C = np.eye(len(p))
        # C = np.diag(p)
        # if p[-1] == 0:
        #     C[-1, -1] = 1
        for i in range(40):
            M = np.zeros((len(p), len(p)))
            b = np.zeros(len(p))
            for j in self.feat.keys():
                up, vp = self.cam.P(xp[j], yp[j])
                lp = self.lvl[j].Interp(up, vp)
                dPudp, dPvdp = self.cam.dPdp(xp[j], yp[j])
                ldxr, ldyr = self.lvl[j].InterpGrad(up, vp)
                dPdl = np.diag(ldxr) @ dPudp + np.diag(ldyr).dot(dPvdp)
                M += C.T.dot(dPdl.T.dot(dPdl.dot(C)))
                b += C.T.dot(dPdl.T.dot(lp))
            dp = C.dot(np.linalg.solve(M, -b))
            p += 0.8 * dp
            self.cam.set_p(p)
            err = np.linalg.norm(dp) / np.linalg.norm(p)
            print("Iter # %2d | disc=%2.2f %% | dU/U=%1.2e"
                % (i + 1, np.mean(lp) / max(self.f.pix.shape) * 100, err))
            if err < 1e-5:
                break
        print("p = np.array([%f, %f, %f, %f])" % (p[0], p[1], p[2], p[3]))
        return self.cam

    def SavePoints(self, filename):
        points = {'ptsi': self.ptsi, 'ptsm': self.ptsm, 'feat': self.feat}
        np.savez(filename, **points)
        print("Writing file %s.npz" % filename)

    def LoadPoints(self, filename):
        params = dict(np.load(filename, allow_pickle=True))
        self.ptsi = params['ptsi'].tolist()
        self.ptsm = params['ptsm'].tolist()
        self.feat = params['feat'].tolist()
        self.nfeat = len(self.feat.keys())
        print("File %s.npz loaded" % filename)