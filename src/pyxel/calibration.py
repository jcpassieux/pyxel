# -*- coding: utf-8 -*-
"""
Created on Thu May 30 08:50:27 2024

@author: passieux
"""

import cv2  # pip install opencv-python
import os
import numpy as np
import matplotlib.pyplot as plt
import svg  # pip install svg.py
from matplotlib.patches import Rectangle
from datetime import datetime
import imageio
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


def ProgressBar(percent):
    width = 40
    left = width * percent // 100
    right = width - left
    tags = "█" * int(np.round(left))
    spaces = " " * int(np.round(right))
    percents = f"{percent:.0f}%"
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)


def Vects2Matrix(r, t):
    R, _ = cv2.Rodrigues(r)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, [-1]] = t
    return T


def Matrix2Vects(T):
    R = T[:3, :3]
    t = T[:3, [-1]]
    r, _ = cv2.Rodrigues(R)
    return r, t


def InverseTransformation(arg0, arg1=None):
    """
    Inverse a transformation (Rotation + Translation)
    Input either as a Matrix T or as two vectors r and t
    """
    as_matrix = True
    if arg1 is not None:
        as_matrix = False
        T = Vects2Matrix(arg0, arg1)
        r = arg0
        t = arg1
    else:
        T = arg0.copy()
        r, t = Matrix2Vects(T)
    by_inversion = False
    if by_inversion:
        Ti = np.linalg.inv(T)
        ti = Ti[:3, [-1]]
        ri, _ = cv2.Rodrigues(Ti[:3, :3])
    else:
        ri = -r
        ti = -T[:3, :3].T@t
    if as_matrix:
        return Vects2Matrix(ri, ti)
    else:
        return ri, ti


def ComposeTransformation(r1, t1, r2, t2):
    """
    Compose two transformations (Rotation + Translation)
    Input as two couples of (r, t) vectors
    (just a product of matrices otherwise)
    """
    R1, _ = cv2.Rodrigues(r1)
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, [-1]] = t1
    R2, _ = cv2.Rodrigues(r2)
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, [-1]] = t2
    T = T1@T2
    tnew = T[:3, [-1]]
    rnew, _ = cv2.Rodrigues(T[:3, :3])
    return rnew, tnew


# %%
# otherwise https://github.com/opencv/opencv/blob/4.x/doc/pattern_tools/gen_pattern.py


class Board():
    def __init__(self, board_size, board_type='chess', board_step=1, ratio=0.25):
        """
        Parameters
        ----------
        board_size : TUPLE or LIST
            DESCRIPTION. Number of squares or circles in each direction.
        board_type : STRING, optional
            DESCRIPTION.
            'chess' chessboard (DEFAULT)
            'circles' circles grid symmetric,
            'acircles' circles grid asymmetric
            'circles_vic' circles grid symmetric from Correlated Solutions
        board_step : FLOAT, optional
            DESCRIPTION. size of the step in mm. The default is 1.

        """
        self.type = board_type
        self.size = np.array(board_size)
        self.step = float(board_step)
        self.ratio = ratio
        if self.type == 'circles_vic':
            self.rmin = 10
            self.rmax = 100

    def GetObjPoints(self):
        """
        Build the coordinates of the 3D points in the coordinate system
        of the board (z=0)

        Returns
        -------
        NUMPY.NDARRAY (FLOAT32 required for opencv calibration tool)
            DESCRIPTION.
            coordinates of the grid points

        """
        if self.type == 'chess':
            x = np.arange(self.size[0]) * self.step
            y = np.arange(self.size[1]) * self.step
            X, Y = np.meshgrid(x, y)
            pts = np.c_[X.ravel(), Y.ravel(), 0*X.ravel()]
        elif self.type == 'circles':
            # x = np.arange(self.size[1]) * self.step
            # y = np.arange(self.size[0]) * self.step
            x = np.arange(self.size[0]) * self.step
            y = np.arange(self.size[1]) * self.step
            X, Y = np.meshgrid(x, y)
            pts = np.c_[X.ravel(), Y.ravel(), 0*X.ravel()]
        elif self.type == 'circles_vic':
            x = np.arange(self.size[0]) * self.step
            y = np.arange(self.size[1]) * self.step
            X, Y = np.meshgrid(x, y)
            pts = np.c_[X.ravel(), Y.ravel(), 0*X.ravel()]
        elif self.type == 'acircles':
            g = []
            for x in range(0, self.size[1]):
                for y in range(0, self.size[0]):
                    cx = (2 * x * self.step) + (y % 2)*self.step
                    cy = y * self.step
                    g += [[cx, cy]]
            pts = np.array(g)
        else:
            print('ERROR unknown pattern type')
        return pts.astype('float32')

    def Plot(self):
        """
        Plots the calibration board in matplotlib
        """
        fig, ax = plt.subplots()
        if 'circles' in self.type:
            pts = self.GetObjPoints()
            ax.plot(pts[:, 0], pts[:, 1], 'ko')
            ax.axis('equal')
        elif self.type == 'chess':
            nx = int(np.ceil((self.size[0]+1) / 2))
            ny = self.size[1] + 1
            rm = self.size[0] % 2 == 0
            for j in range(ny):
                for i in range(nx - (j % 2)*rm):
                    x = 2*i*self.step + (j % 2)*self.step
                    y = j*self.step
                    ax.add_patch(Rectangle((x, y), width=self.step, height=self.step, facecolor='k'))
            ax.set_xlim(-self.step, nx*self.step)
            ax.set_ylim(-self.step, ny*self.step)
            ax.invert_yaxis()
            ax.axis('equal')
        else:
            print('ERROR unknown pattern type')
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()


    def SVGPrint(self, filename='calibration_pattern.svg', paper_size=[210, 297]):
        """
        Create a SVG file to print the calibration board on a printer.

        Parameters
        ----------
        filename : STRING, optional
            DESCRIPTION. The default is 'calibration_pattern.svg'.
        paper_size : LIST or NUMPY.ARRAY, optional
            DESCRIPTION. The default is [210, 297] equiv. to A4 paper format
        """
        if self.type == 'chess':
            board_dim = (np.array(self.size)+1) * self.step
            start = np.array(paper_size)/2 - board_dim/2
            nx = int(np.ceil((self.size[0]+1) / 2))
            ny = self.size[1] + 1
            rm = self.size[0] % 2 == 0
            elements = []
            for j in range(ny):
                for i in range(nx - (j%2)*rm):
                    elements += [svg.Rect(
                        x = start[0] + 2*i*self.step + (j%2)*self.step,
                        y = start[1] + j*self.step,
                        width = self.step,
                        height = self.step,
                        fill="black"), ]
        elif 'circles' in self.type:
            radius = self.ratio * self.step
            pts = self.GetObjPoints()
            board_dim = np.array([np.max(pts[:, 0])-np.min(pts[:, 0]),
                                  np.max(pts[:, 1])-np.min(pts[:, 1])])
            start = (np.array(paper_size) - board_dim)/2
            elements = []
            for pti in pts:
                elements += [svg.Circle(
                    cx = start[0] + pti[0],
                    cy = start[1] + pti[1],
                    r = radius,
                    fill="black"), ]
        else:
            print('ERROR unknown pattern type')
        text = '%s %dx%d step %1.0f mm' % (self.type, self.size[0], self.size[1], self.step)
        elements += [svg.Text(x=5, y=5, text=text, opacity=0.2, font_size=3, font_style='italic')]

        canvas = svg.SVG(width=paper_size[0], height=paper_size[1],
                         elements=elements)
        fichier = open(filename, "w")
        fichier.write(canvas.as_str())
        fichier.close()

# %%

def PlotLocalCoordinates(board, img, points):
    points = np.int32(points)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[3,0]), (255,0,0), 3, tipLength=0.05)
    cv2.arrowedLine(img, tuple(points[0,0]), tuple(points[board.size[0]*3,0]), (255,0,0), 3, tipLength=0.05)
    cv2.circle(img, tuple(points[0,0]), 8, (0,255,0), 3)
    cv2.putText(img, '0,0', (points[0,0,0]-35, points[0,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'X', (points[3,0,0]-25, points[3,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, 'Y', (points[board.size[0]*3,0,0]-25, points[board.size[0]*3,0,1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

# code https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv
# theory https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
# help asymmetric https://longervision.github.io/2017/03/18/ComputerVision/OpenCV/opencv-internal-calibration-circle-grid/


def ImagesFromMovie(moviename, frames=None):
    """
    Extract Images from a movie

    Parameters
    ----------
    moviename : STRING
    frames : NUMPY ARRAY of TYPE INT
        only the wanted frames

    Returns
    -------
    dir0 : STRING
        directory where the images are extracted

    """
    vid = imageio.get_reader(moviename, 'ffmpeg')
    if frames is None:
        nframes = vid.count_frames()
        frames = np.arange(nframes)
    else:
        nframes = len(frames)
    dir0 = moviename[:moviename.rfind('.')]
    dir0 = os.path.join("imgs", dir0)
    if not os.path.isdir(dir0):
        os.makedirs(dir0)
    print('Extracting images from movie file...')
    for count in range(len(frames)):
        ProgressBar(100*(count+1)/nframes)
        frame = vid.get_data(frames[count])
        # ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(dir0, 'img_%04d.tif' % frames[count])
        cv2.imwrite(output_path, gray)
    return dir0

def BoardRotation(pts, board):
    """
    Rotates the grid points such that
    For square boards: Rotates the grid points by 90° until the first point is the upper left corner
    For rectangular boards: rotates the grid by 180° and find X minimal at the top
    
    Parameters
    ----------
    pts : NUMPY.ARRAY
        list of grid points size (npts, 1, 2) 
    board : BOARD

    Returns
    -------
    pts : TYPE
        DESCRIPTION.

    """
    if board.size[0] == board.size[1]:
        X = pts[:, 0, 0].reshape(board.size[::-1])
        Y = pts[:, 0, 1].reshape(board.size[::-1])
        it = 0
        while np.argmin(np.sqrt(X**2+Y**2)):
            X = np.rot90(X, k=1)
            Y = np.rot90(Y, k=1)
            it += 1
            if it > 3:
                raise Exception('Too many attempts for board rotations for Image %d...')
        pts[:, 0, 0] = X.ravel()
        pts[:, 0, 1] = Y.ravel()
    else:
        X = pts[:, 0, 0].reshape(board.size[::-1])
        Y = pts[:, 0, 1].reshape(board.size[::-1])
        # plt.imshow(np.sqrt(X**2+Y**2))
        # plt.colorbar()
        # img = self.GetImage(self.img_frame_id[i])
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = cv2.drawChessboardCorners(img, board.size, self.imgpoints[i], True)
        # plt.imshow(img)
        X180 = np.rot90(X, k=2)
        Y180 = np.rot90(Y, k=2)
        if np.argmin(X180**2+Y180**2) < np.argmin(X**2+Y**2):
            X = X180
            Y = Y180
        pts[:, 0, 0] = X.ravel()
        pts[:, 0, 1] = Y.ravel()
    return pts


class Camera:
    def __init__(self, image_dir=None):
        self.image_dir = image_dir
        if image_dir:
            self.LoadImages(image_dir)
        self.imgpoints = None
        self.img_frame_id = []
        self.params = dict()

    def LoadImages(self, image_dir):
        self.image_dir = image_dir
        if image_dir:
            print('%3d Images found in directory %s' % (len(os.listdir(image_dir)), image_dir))
            self.SortImageNames()

    def SortImageNames(self):
        path = self.image_dir
        imagelist = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        # imagelist = sorted(os.listdir(path))
        ftype = []
        lengths = []
        for name in imagelist:
            ftype.append(name.split('.')[-1])
            lengths.append(len(name))
        lengths = sorted(list(set(lengths)))
        ImageNames = []
        for ll in lengths:
            for name in imagelist:
                if len(name) == ll:
                    ImageNames.append(os.path.join(path, name))
        self.filenames = ImageNames

    def Plot(self, imnum=0, opt=False):
        """
        Plot one image of the sequence

        Parameters
        ----------
        imnum : TYPE, optional
            DESCRIPTION. The default is 0.

        """
        img = self.GetImage(imnum, opt)
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.axis('off')

    def GetImage(self, imnum, opt=False):
        img = cv2.imread(self.filenames[imnum])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # alpha = 2. # Contrast control (1.0-3.0)
        # beta = 60 # Brightness control (0-100)
        # adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if not opt:
            opt = dict()
        for ik in opt.keys():
            if ik == 'rescale':
                fmin = opt[ik][0]
                fmax = opt[ik][1]
                img = 1.0 * (img-fmin)/(fmax-fmin) * 255
                img[np.where(img > 255)] = 255
                img[np.where(img < 0)] = 0
                img = np.round(img).astype('uint8')
            elif ik == 'median':
                med_blur = opt[ik]
                if med_blur:
                    med_blur = max(med_blur, 1)
                    img = cv2.medianBlur(img, med_blur)
            elif ik == 'gaussian':
                gau_blur = opt[ik]
                if gau_blur:
                    # gau_blur[0] = max(gau_blur[0], 1)
                    # gau_blur[1] = max(gau_blur[1], 1)
                    img = cv2.GaussianBlur(img, gau_blur, 0)
            elif ik == 'adaptive_threshold':
                ad_thrs = opt[ik]
                if ad_thrs:
                    img = cv2.adaptiveThreshold(img, 255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        ad_thrs[0], ad_thrs[1])
            elif ik == 'morphology_ex':
                mor_ex = opt[ik]
                if mor_ex is not None:
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, mor_ex)
        return img

    def findChessboardCornersVIC(self, board, imnum, opt):
        d_min = board.rmin*2
        d_max = board.rmax*2
        rad = int(board.rmin)
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 3.14*d_min**2/4
        params.maxArea = 3.14*d_max**2/4
        params.filterByColor = False
        params.filterByConvexity = False
        params.minCircularity = 0.2
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        detector = cv2.SimpleBlobDetector_create(params)
        gray = self.GetImage(imnum, opt)
        keypoints = detector.detect(gray)
        gray = self.GetImage(imnum, dict())
        x, y = np.meshgrid(np.arange(-rad, rad).astype(int),
                           np.arange(-rad, rad).astype(int))
        x = x.ravel()
        y = y.ravel()
        rep, = np.where(x**2+y**2 < (rad*0.8)**2)
        x = x[rep]
        y = y[rep]
        stds = np.zeros(len(keypoints))
        i = 0
        for kpt in keypoints:
            xx = x + int(kpt.pt[0])
            yy = y + int(kpt.pt[1])
            coul = gray[yy, xx]
            stds[i] = np.std(coul)
            i += 1
        X = np.array([list(kpt.pt) for kpt in keypoints])
        rep = np.argsort(stds)[-3:]
        X3 = X[rep[[0, 1, 2, 0]]]
        v3 = np.diff(X3, axis=0)  # 3 vectors of the triangle
        l3 = np.linalg.norm(v3, axis=1)  # 3 triangle edge size
        lsort = np.argsort(l3)  # sort by edge size
        if lsort[-1] == 0:
            # if the larger size is 0, the right angle is 2
            Xb = X3[2]
            Vb = np.array([X3[0] - Xb, X3[1] - Xb])
        elif lsort[-1] == 1:
            # if the larger size is 1, the right angle is 0
            Xb = X3[0]
            Vb = np.array([X3[1] - Xb, X3[2] - Xb])
        else:
            # if the larger size is 2, the right angle is 1
            Xb = X3[1]
            Vb = np.array([X3[0] - Xb, X3[2] - Xb])
        if np.diff(np.linalg.norm(Vb, axis=1))[0] < 1:
            Vb = Vb[::-1]
        Vb = Vb/np.linalg.norm(Vb, axis=1)[np.newaxis].T
        # the first dimension is along the smaller size.
        if np.diff(board.size)[0] < 0:
            inner_size = board.size[::-1].tolist()
        else:
            inner_size = board.size.tolist()

        def ClosestPoint(X, x):
            rep = np.argsort((X[:, 0] - x[0])**2 + (X[:, 1] - x[1])**2)[0]
            # dist = (X[rep, 0] - x[0])**2 + (X[rep, 1] - x[1])**2
            return X[rep]
        detected_points = np.zeros([2, ] + inner_size)
        step_y = l3[lsort[1]] / (inner_size[1]-1)
        step_x = l3[lsort[0]] / (inner_size[0]-1)
        yold = Xb.copy()
        vec_y = step_y * Vb[1]
        for iy in range(inner_size[1]):
            if iy > 0:
                xnew = yold + vec_y
                xnew = ClosestPoint(X, xnew)
                detected_points[:, 0, iy] = xnew
                vec_y = xnew - yold
                yold = xnew.copy()
                xold = yold.copy()
            else:
                xold = yold.copy()
                detected_points[:, 0, 0] = xold
            vec_x = step_x * Vb[0]
            for ix in range(1, inner_size[0]):
                xnew = xold + vec_x
                xnew = ClosestPoint(X, xnew)
                vec_x = xnew - xold
                xold = xnew.copy()
                detected_points[:, ix, iy] = xnew
        imagepts = np.zeros([np.prod(inner_size), 1, 2])
        imagepts[:, 0, 0] = detected_points[0, :, :].ravel()
        imagepts[:, 0, 1] = detected_points[1, :, :].ravel()
        return True, imagepts

    def DetectPoints_i(self, board, imnum, opt=False):
        """
        Detect board points in image i

        Parameters
        ----------
        board :
            CALIBRATION BOARD
        imnum : INT
            FRAME NUMBER.
        opt : DICT, optional
            DESCRIPTION.
            'median': 5 for a median filter
            'gaussian': [5, 5] for a gaussian filter
            'adaptive_threshold'
            'morphology_ex'
            'cb_clustering' BOOL

        Returns
        -------
        res : BOOL
            True if the points were detected.
        pts : NUMPY.ARRAY
            GRID POINTS LOCATION

        """
        img = self.GetImage(imnum, opt)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if not opt:
            opt = dict()
        if board.type == 'chess':
            # ##### CHESSBOARDS ##### #
            res, pts = cv2.findChessboardCorners(img, board.size)
            if res:
                pts = cv2.cornerSubPix(img, pts, (4, 4), (-1, -1), criteria)

        elif board.type == 'circles_vic':
            # ##### CORRELATED BOARDS ##### #
            res, pts = self.findChessboardCornersVIC(board, imnum, opt)

        elif board.type in ['circles', 'acircles']:
            # ##### CIRCLE GRIDS ##### #
            if board.type == 'circles':
                flags = cv2.CALIB_CB_SYMMETRIC_GRID
            elif board.type == 'acircles':
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            if 'cb_clustering' in opt.keys():
                if opt['cb_clustering']:
                    flags += cv2.CALIB_CB_CLUSTERING
            res, pts = cv2.findCirclesGrid(img, board.size, flags=(flags))
            if not res:
                res, pts = cv2.findCirclesGrid(img, board.size[::-1], flags=(flags))
            if res:
                pts = cv2.cornerSubPix(img, pts, (4, 4), (-1, -1), criteria)
        else:
            raise Exception('ERROR unknown pattern type: %s' % board.type)
        return res, pts

    def DetectPoints_circles(self, board, img, opt=False):
        """
        Detect board points in one image

        Parameters
        ----------
        board :
            CALIBRATION BOARD
        img : INT or NUMPY.ARRAY
            FRAME NUMBER or (preprocessed) graylevel IMAGE
        opt : see DetectPoints

        """
        if type(img) is int:
            img = self.GetImage(img, opt)
        if board.type not in ['circles', 'acircles']:
            raise Exception('Circles grid required, but %s given' % board.type)
        if board.type == 'circles':
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
        elif board.type == 'acircles':
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
        if 'cb_clustering' in opt.keys():
            if opt['cb_clustering']:
                flags += cv2.CALIB_CB_CLUSTERING
        res, pts = cv2.findCirclesGrid(img, board.size, flags=(flags))
        if not res:
            res, pts = cv2.findCirclesGrid(
                img, board.size[::-1], flags=(flags))
        if res:
            cvcode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
            if 'maxiter' not in opt.keys():
                maxiter = 30
            else:
                maxiter = opt['maxiter']
            if 'eps' not in opt.keys():
                eps = 0.001
            else:
                eps = opt['eps']
            criteria = (cvcode, maxiter, eps)
            pts = cv2.cornerSubPix(img, pts, (4, 4), (-1, -1), criteria)
            if opt['cb_orientation']:
                pts = BoardRotation(pts, board)
        return res, pts

    def DetectPoints_chess(self, board, img, opt={}):
        """
        Detect board points in one image

        Parameters
        ----------
        board :
            CALIBRATION BOARD
        img : INT or NUMPY.ARRAY
            FRAME NUMBER or (preprocessed) graylevel IMAGE
        opt : see DetectPoints

        """
        if type(img) is int:
            img = self.GetImage(img, opt)
        if board.type != 'chess':
            raise Exception('Chessboard required, but %s given' % board.type)

        res, pts = cv2.findChessboardCorners(img, board.size)
        if res:
            cvcode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
            if 'maxiter' not in opt.keys():
                maxiter = 30
            else:
                maxiter = opt['maxiter']
            if 'eps' not in opt.keys():
                eps = 0.001
            else:
                eps = opt['eps']
            criteria = (cvcode, maxiter, eps)
            pts = cv2.cornerSubPix(img, pts, (3, 3), (-1, -1), criteria)
            if 'cb_orientation' in opt.keys():
                if opt['cb_orientation']:
                    pts = BoardRotation(pts, board)
            else:
                pts = BoardRotation(pts, board)
        return res, pts

    def DetectPoints_circlesvic(self, board, imnum, opt=None):
        """
        Detect board points in one image using Correlated like boards.
        Detects the local coord sys in order to orient the board in space
        allow for large rotations of the board during the calibration (for JN!)

        Parameters
        ----------
        board :
            CALIBRATION BOARD
        img : INT or NUMPY.ARRAY
            FRAME NUMBER or (preprocessed) graylevel IMAGE
        opt : see DetectPoints

        """
        img = self.GetImage(imnum, opt)
        if board.type != 'circles':
            raise Exception('Circles grid required, but %s given' % board.type)
        if not opt['cb_clustering']:
            print('CB CLUSTERING REQUIRED')
        # CB clustering required
        flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        res, pts = cv2.findCirclesGrid(img, board.size, flags=(flags))
        if not res:
            res, pts = cv2.findCirclesGrid(
                img, board.size[::-1], flags=(flags))
        if res:
            cvcode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
            if 'maxiter' not in opt.keys():
                maxiter = 30
            else:
                maxiter = opt['maxiter']
            if 'eps' not in opt.keys():
                eps = 0.001
            else:
                eps = opt['eps']
            criteria = (cvcode, maxiter, eps)
            pts = cv2.cornerSubPix(img, pts, (4, 4), (-1, -1), criteria)
            if opt['cb_orientation']:
                # open again the image but without filtering
                gray = self.GetImage(imnum, dict())
                X = pts[:, 0, 1].reshape(board.size[::-1]).astype('int64')
                Y = pts[:, 0, 0].reshape(board.size[::-1]).astype('int64')
                # id_tab = np.arange(pts.shape[0]).reshape(board.size[::-1])
                # cam0.Plot(i)
                # plt.scatter(Y, X, c=id_tab)
                # evaluation of the graylevels of the centers
                gray_pts = gray[X, Y]
                # define the threshold
                trh = (np.max(gray_pts) + np.min(gray_pts)) * 0.5
                # detection of the 3 white dots
                repx, repy = np.where(gray_pts > trh)
                good = np.array_equal(repy, np.array(
                    [2, board.size[0]-3, board.size[0]-3])) * \
                    np.array_equal(repx, np.array([2, 2, board.size[1]-3]))
                if not good:
                    # rotation by 180 degree
                    pts[:, 0, 0] = np.rot90(pts[:, 0, 0].reshape(
                        board.size[::-1]), k=2).ravel()
                    pts[:, 0, 1] = np.rot90(pts[:, 0, 1].reshape(
                        board.size[::-1]), k=2).ravel()
        return res, pts

    def DetectPoints(self, board, opt=None):
        """
        Detect board points in the whole image sequence

        Parameters
        ----------
        board : BOARD
            DESCRIPTION.
            
        opt : DICT, optional
            'median': 5 for a median filter
            'gaussian': [5, 5] for a gaussian filter
            'adaptive_threshold'
            'morphology_ex' 
            'cb_clustering' : special algo for circles grid detection. More robust to perspective 
                              distortions but much more sensitive to background clutter. 
            'method' : 'chess' : chessboards using cv2.findChessboardCorners
                        'circles' : symmetric circle grids using cv2.findCirclesGrid
                        'acircles' : asymmetric circle grids using cv2.findCirclesGrid
                        'circlesvic' : Correlated-like grids with dots in 3 circles for local csys. FOR JN!!
                        'circlesvic2' : Like 'circlevic' but using only the central circles
                        The default is None, automatic choice based on the board type.
            'cb_orientation' : if True, corrects the orientation of the board such that the origin
                               of the board coord system is in the top left corner.
            'eps' : cv2.TERM_CRITERIA_EPS, default 0.001
            'maxiter' : cv2.TERM_CRITERIA_MAX_ITER, default 30
        """

        print('Detecting Points in Images...')

        # setting default options if not provided
        if opt is None:
            opt = dict()

        if 'cb_orientation' not in opt.keys():
            opt['cb_orientation'] = True
        if 'cb_clustering' not in opt.keys():
            opt['cb_clustering'] = True
        if 'eps' not in opt.keys():
            opt['eps'] = 0.001
        if 'maxiter' not in opt.keys():
            opt['maxiter'] = 30
        if 'method' not in opt.keys():
            opt['method'] = board.type

        imgpoints = []
        nimg = len(self.filenames)
        images_working = []
        for imnum in range(nimg):
            ProgressBar(100*(imnum+1)/nimg)
            img = self.GetImage(imnum, opt)
            if opt['method'] in ['circles', 'acircles']:
                res, pts = self.DetectPoints_circles(board, img, opt)
            elif opt['method'] == 'circlesvic':
                res, pts = self.DetectPoints_circlesvic(board, imnum, opt)
            elif opt['method'] == 'circlesvic2':
                res, pts = self.DetectPoints_circlesvic2(board, img, opt)
            elif opt['method'] == 'chess':
                res, pts = self.DetectPoints_chess(board, img, opt)
            else:
                raise Exception('Unknown detection method')
            if res:
                imgpoints.append(pts.astype('float32'))
                images_working += [imnum,]
        self.imgpoints = imgpoints
        self.img_frame_id = np.array(images_working)
        print(' > Score: %3.0f %%' % (len(self.img_frame_id)/nimg*100, ))

    def PlotDetectedPoints_i(self, board, imnum):
        img = self.GetImage(imnum)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        found = np.where(imnum == self.img_frame_id)[0]
        if len(found):
            points = self.imgpoints[found[0]]
            img = cv2.drawChessboardCorners(img, board.size, points, True)
        org = (50, 50)
        string = '%d' % imnum
        cv2.putText(img, string, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow('animation', cv2.WINDOW_NORMAL)
        cv2.imshow('animation', img)

    def PlotDetectedPoints(self, board, imnum=False, wait=1):
        """
        Plot the detected board points in one image of the sequence

        Parameters
        ----------
        board : BOARD
            DESCRIPTION.
        imnum : INT, optional
            DESCRIPTION. The default is 0.

        """
        if self.imgpoints is None:
            raise Exception('First detect points using CAMERA.DetectPoints')
        if imnum is False:
            for i in range(len(self.filenames)):
                self.PlotDetectedPoints_i(board, i)
                if cv2.waitKey(wait) == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else:
            self.PlotDetectedPoints_i(board, imnum)

    def SaveDetectedPoints(self, filename=None):
        if not filename:
            filename = datetime.now().strftime("detected_pts_%y%m%d-%H%M%S-%f")
        np.savez(filename, IP=self.imgpoints, IF=self.img_frame_id)
        print("Writing file %s.npz" % filename)

    def LoadDetectedPoints(self, filename):
        self.imgpoints = np.load(filename)['IP']
        self.img_frame_id = np.load(filename)['IF']

    def PlotLocalCoordinate_i(self, board, imnum):
        img = self.GetImage(imnum)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        found = np.where(imnum == self.img_frame_id)[0]
        if len(found):
            points = self.imgpoints[found[0]]
            img = PlotLocalCoordinates(board, img, points)
        org = (50, 50)
        string = '%d' % imnum
        cv2.putText(img, string, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.namedWindow('anima2', cv2.WINDOW_NORMAL)
        cv2.imshow('anima2', img)

    def PlotLocalCoordinate(self, board, imnum=False, wait=1):
        """
        Plot the local coordinate system of the board in one image of the
        sequence.

        Parameters
        ----------
        board : BOARD
            DESCRIPTION.
        imnum : INT, optional
            DESCRIPTION. Image number. The default is 0.
        """
        if self.imgpoints is None:
            raise Exception('First detect points using CAMERA.DetectPoints')
        if imnum is False:
            for i in range(len(self.filenames)):
                self.PlotLocalCoordinate_i(board, i)
                if cv2.waitKey(wait) == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else:
            self.PlotLocalCoordinate_i(board, imnum)

    def Calibrate(self, board, dist=True):
        """
        Perform the calibration of a Camera given the coordinates of grid
        points found in a set of images

        Parameters
        ----------
        board : BOARD
        """
        if self.imgpoints is None:
            self.DetectPoints(board)
        CameraParams = {}
        img = cv2.cvtColor(cv2.imread(self.filenames[0]), cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        objpoints = board.GetObjPoints()
        objp = []
        for i in range(len(self.imgpoints)):
            objp.append(objpoints)
        if dist:
            res, K, D, R, T = cv2.calibrateCamera(objp, self.imgpoints, (w, h), None, None, flags=0)
        else:
            res, K, D, R, T = cv2.calibrateCamera(objp, self.imgpoints, (w, h), (0, 0, 0, 0, 0), None, flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST)
        Rmtx = []
        Tmtx = []
        k = 0
        for r in R:
            Rmtx.append(cv2.Rodrigues(r)[0])
            Tmtx.append(np.vstack((np.hstack((Rmtx[k],T[k])),np.array([0,0,0,1]))))
            k += 1
        newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        if np.sum(roi) == 0:
            roi = (0, 0, w-1, h-1)

        CameraParams['Intrinsic'] = K
        CameraParams['Distortion'] = D
        CameraParams['DistortionROI'] = roi
        CameraParams['DistortionIntrinsic'] = newK
        CameraParams['RotVector'] = R
        CameraParams['TransVector'] = T
        CameraParams['RotMatrix'] = Rmtx
        CameraParams['Extrinsic'] = Tmtx

        # Estimating reprojection error
        imgp = np.array(self.imgpoints)
        imgp = imgp.reshape((imgp.shape[0], imgp.shape[1], imgp.shape[3]))
        N = imgp.shape[0]
        imgpNew = []
        for i in range(N):
            temp, _ = cv2.projectPoints(objpoints, R[i], T[i], K, D)
            imgpNew.append(temp.reshape((temp.shape[0], temp.shape[2])))
        imgpNew = np.array(imgpNew)
        err = []
        for i in range(N):
            err.append(imgp[i] - imgpNew[i])
            # plt.figure()
            # plt.plot(imgp[i][:, 0], imgp[i][:, 1], 'ro')
            # plt.plot(imgpNew[i][:, 0], imgpNew[i][:, 1], 'y+')
        err = np.array(err)

        def RMSE(err):
            return np.sqrt(np.mean(np.sum(err**2, axis=1)))

        errall = np.copy(err[0])
        rmsePerView = [RMSE(err[0])]
        for i in range(1, N):
            errall = np.vstack((errall, err[i]))
            rmsePerView.append(RMSE(err[i]))
        rmseAll = RMSE(errall)

        CameraParams['Imgpoints'] = self.imgpoints
        CameraParams['Errors'] = rmsePerView
        CameraParams['MeanError'] = rmseAll
        self.params = CameraParams
        # self.PlotParams()

    def PlotParams(self):
        """
        Plots calibrated camera parameters.
        """
        np.set_printoptions(suppress=True, precision=5)
        if 'Intrinsic' in self.params.keys():
            print('Intrinsic Matrix:')
            print(self.params['Intrinsic'])
        if 'Distortion' in self.params.keys():
            print('\nDistortion Parameters:')
            print(self.params['Distortion'])
        if 'Extrinsic' in self.params.keys():
            print('\nExtrinsic Matrix from 1.Image:')
            print(self.params['Extrinsic'][0])
        if 'Errors' in self.params.keys():
            print('Mean Reprojection Error:  {:.4f}'.format(self.params['MeanError']))
            max_err = np.max(self.params['Errors'])
            print('Max Reprojection Error: {:.4f}'.format(max_err))

    def PlotParamsVIC(self, num=None):
        """
        Plot Calibrated Stereo Parameters like VIC
        """
        if num is None:
            print("\nCamera intrinsics:")
        else:
            print("\nCamera %d intrinsics:" % num)
        [[fx, _, cx],
         [_, fy, cy],
         [_, _, _]] = self.params["Intrinsic"]
        print(f"Center (X): {cx:.3f}")
        print(f"Center (Y): {cy:.3f}")
        print(f"Focal Length (X): {fx:.3f}")
        print(f"Focal Length (Y): {fy:.3f}")
        [[kappa1, kappa2, p1, p2, kappa3]] = self.params["Distortion"]
        print(f"Kappa 1: {kappa1:.3f}")
        print(f"Kappa 2: {kappa2:.3f}")
        print(f"p1: {p1:.3f}")
        print(f"p2: {p2:.3f}")
        print(f"Kappa 3: {kappa3:.3f}")

    def PlotProjErrors(self):
        plt.bar(self.img_frame_id, np.array(self.params['Errors']), color='k')
        plt.title('Reprojection errors')
        plt.xlabel('frame number')

    def RemoveImgHighErrors(self, err_max=1):
        rep, = np.where(np.array(self.params['Errors']) < err_max)
        n_remove = len(self.params['Errors'])-len(rep)
        ratio = n_remove/len(self.params['Errors'])
        print('Removing %d Images [%2.0f%%]' % (n_remove, 100*ratio))
        self.img_frame_id = self.img_frame_id[rep]
        self.imgpoints = list(np.array(self.imgpoints)[rep])
        self.params['Errors'] = list(np.array(self.params['Errors'])[rep])

    def SaveParams(self, filename=None):
        if not filename:
            filename = datetime.now().strftime("cam_param_%y%m%d-%H%M%S-%f")
        np.savez(filename, **self.params)
        print("Writing file %s.npz" % filename)

    def LoadParams(self, filename):
        self.params = dict(np.load(filename))
        self.imgpoints = self.params['Imgpoints']

    def UndistortPoints(self, uv):
        K = self.params['Intrinsic']
        dist = self.params['Distortion']
        # uvd = cv2.undistortPoints(uv, K, dist, None, K)
        opt = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.03)
        uvd = cv2.undistortPointsIter(uv, K, dist, None, K, opt)
        return uvd

    def UndistortImage(self, img):
        K = self.params['Intrinsic']
        dist = self.params['Distortion']
        return cv2.undistort(img, K, dist)
# %%


class StereoRig():
    def __init__(self, cam0, cam1):
        """
        Constructor of the stereo Rig

        Parameters
        ----------
        cam0 : CAMERA
            DESCRIPTION.
        cam1 : CAMERA
            DESCRIPTION.

        """
        self.cam0 = cam0
        self.cam1 = cam1
        self.params = {}
        self.params['Rotation'] = np.zeros((3, 1))
        self.params['Translation'] = np.zeros((3, 1))

    def StereoCalibration(self, board, fix_intrinsic=False):
        """
        Performs the stereo calibration given two calibrated camera and their
        detected grid points

        Parameters
        ----------
        board : BOARD
        """
        k1 = self.cam0.params['Intrinsic']
        d1 = self.cam0.params['Distortion']
        k2 = self.cam1.params['Intrinsic']
        d2 = self.cam1.params['Distortion']
        _, ind1, ind2 = np.intersect1d(
            self.cam0.img_frame_id, self.cam1.img_frame_id, return_indices=True)
        imgpoints1 = []
        imgpoints2 = []
        for i in range(len(ind1)):
            # computing ideal points (without distortion) from image points
            imgpoints1 += [self.cam0.UndistortPoints(
                self.cam0.imgpoints[ind1[i]])]
            imgpoints2 += [self.cam1.UndistortPoints(
                self.cam1.imgpoints[ind2[i]])]
            # imgpoints1 += [self.cam0.imgpoints[ind1[i]]]
            # imgpoints2 += [self.cam1.imgpoints[ind2[i]]]
        img = self.cam0.GetImage(0)
        h, w = img.shape
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        flags = 0
        if fix_intrinsic:
            flags |= cv2.CALIB_FIX_INTRINSIC
        else:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        objpoints = board.GetObjPoints()
        objp = []
        for i in range(len(ind1)):
            objp.append(objpoints)
        (ret, K1, D1, K2, D2, R, t, E, F) = cv2.stereoCalibrate(objp, imgpoints1,
               imgpoints2, k1, d1, k2, d2, (w, h), criteria=criteria, flags=flags)
        
        T = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))

        self.params['Transformation'] = T
        self.params['Essential'] = E
        self.params['Fundamental'] = F
        self.params['MeanError'] = ret
        # self.PlotParams()

    def PlotParams(self):
        """
        Plot Calibrated Stereo Parameters
        """
        np.set_printoptions(suppress=True, precision=5)
        if 'Transformation' in self.params.keys():
            print('Transformation Matrix:')
            print(self.params['Transformation'])
        if 'Essential' in self.params.keys():
            print('\nEssential Matrix:')
            print(self.params['Essential'])
        if 'Fundamental' in self.params.keys():
            print('\nFundamental Matrix:')
            print(self.params['Fundamental'])
        if 'MeanError' in self.params.keys():
            print('\nMean Reprojection Error:')
            print('{:.6f}'.format(self.params['MeanError']))

    def PlotParamsVIC(self):
        """
        Plot Calibrated Stereo Parameters VIC style
        """
        # self.cam0.PlotParamsVIC(0)
        # self.cam1.PlotParamsVIC(1)
        print("\nExtrinsics: \n\nAngles: \n")
        T = self.params['Transformation']
        R = T[:3, :3]
        r = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        rx, ry, rz = r
        print(f"X: {rx:.3f}")
        print(f"Y: {ry:.3f}")
        print(f"Z: {rz:.3f}")
        print("\nDistances: \n")
        t = T[:3, -1]
        tx, ty, tz = t
        print(f"X [mm]: {tx:.3f}")
        print(f"Y [mm]: {ty:.3f}")
        print(f"Z [mm]: {tz:.3f}")

    def Triangulation(self, pts1, pts2):
        """
        Triangulate 3D points from coordinates of stereo-corresponding points
        in both camera coordinate systems.

        Parameters
        ----------
        pts1 : NUMPY.ARRAY
            DESCRIPTION. One of the dimensions must be equal to 2
        pts2 : NUMPY.ARRAY
            DESCRIPTION. One of the dimensions must be equal to 2

        Returns
        -------
        result : NUMPY.ARRAY
            DESCRIPTION. an array containing the 3 coordinates of the
            triangulated points.

        """
        K1 = self.cam0.params['Intrinsic']
        K2 = self.cam1.params['Intrinsic']
        # transformation between world CSYS and Cam0
        r0 = self.params['Rotation']
        t0 = self.params['Translation']
        T0 = Vects2Matrix(r0, t0)
        # Composition of transformations T0 and T12
        T1 = self.params['Transformation'] @ T0
        projMatr1 = K1 @ T0[:-1]
        projMatr2 = K2 @ T1[:-1]
        switch_col_row = (pts1.shape[1] == 2)
        if switch_col_row:
            pts1 = pts1.T
            pts2 = pts2.T
        pts1 = self.cam0.UndistortPoints(pts1)
        pts2 = self.cam1.UndistortPoints(pts2)
        result = cv2.triangulatePoints(projMatr1, projMatr2, pts1, pts2)
        result = result[:-1, :] / result[-1, :]
        if switch_col_row:
            result = result.T
        return result

    def Projection(self, pts):
        # transformation between World and Cam0
        r0 = self.params['Rotation']
        t0 = self.params['Translation']
        # Transformation between Cam0 and Cam1
        T = self.params['Transformation']
        t = T[:3, [-1]]
        r, _ = cv2.Rodrigues(T[:3, :3])
        r1, t1 = ComposeTransformation(r, t, r0, t0)
        # Projection
        K0 = self.cam0.params['Intrinsic']
        D0 = self.cam0.params['Distortion']
        K1 = self.cam1.params['Intrinsic']
        D1 = self.cam1.params['Distortion']
        uv0, _ = cv2.projectPoints(pts, r0, t0, K0, D0)
        uv1, _ = cv2.projectPoints(pts, r1, t1, K1, D1)
        return uv0, uv1

    def CalibrateExtrinsicFrom3DPoints(self, XYZ, uv0, uv1):
        """
        Calibration of Extrinsic of the Rig (between World CSYS and Cam0)
        from a 3D point cloud.

        Parameters
        ----------
        XYZ : NUMPY ARRAY
            3D Coordinates of the triangulated points
        uv0 : NUMPY ARRAY
            2D coordinates of the projection of the points in cam 0
        uv1 : NUMPY ARRAY
            2D coordinates of the projection of the points in cam 1

        """
        # initialization
        X3d = self.Triangulation(uv0, uv1)
        t = (np.mean(XYZ, axis=0) - np.mean(X3d, axis=0))[:, np.newaxis]
        r = np.zeros((3, 1))

        def residual(p):
            R, _ = cv2.Rodrigues(p[:3, None])
            t = p[3:][:, np.newaxis]
            return (X3d.T - R@XYZ.T - t).ravel()

        params = least_squares(residual, np.append(r, t))
        r = params.x[:3][:, np.newaxis]
        t = params.x[3:][:, np.newaxis]
        # res = residual(np.append(r, t))
        # plt.plot(res)
        rnew, tnew = ComposeTransformation(
            self.params['Rotation'], self.params['Translation'], r, t)
        self.params['Rotation'] = rnew
        self.params['Translation'] = tnew

    def Plot3DBoards(self, cam_centers=False, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        alpha = 1
        _, ind1, ind2 = np.intersect1d(
            self.cam0.img_frame_id, self.cam1.img_frame_id, return_indices=True)
        imgpoints1 = [self.cam0.imgpoints[i] for i in ind1]
        imgpoints2 = [self.cam1.imgpoints[i] for i in ind2]
        for i in range(len(imgpoints1)):
            pts1 = imgpoints1[i][:, 0, :]
            pts2 = imgpoints2[i][:, 0, :]
            pts3d = self.Triangulation(pts1, pts2)
            ax.plot(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], 'o', alpha=alpha)
            alpha = 0.2
        if cam_centers:
            plt.plot(0, 0, 0, 'ko')
            cc = np.linalg.solve(
                self.params['Transformation'], np.array([[0, 0, 0, 1]]).T)
            plt.plot(cc[0], cc[1], cc[2], 'ro')
        aa = plt.axis()
        amax = max(max(aa[1]-aa[0], aa[3]-aa[2]), aa[5]-aa[4])/2
        ax.set_xlim((aa[1]+aa[0])/2-amax, (aa[1]+aa[0])/2+amax)
        ax.set_ylim((aa[3]+aa[2])/2-amax, (aa[3]+aa[2])/2+amax)
        ax.set_zlim((aa[5]+aa[4])/2-amax, (aa[5]+aa[4])/2+amax)
        return ax

    def CheckBoardStep(self, board, imnum=0):
        pts1 = self.cam0.imgpoints[imnum][:, 0, :]
        pts2 = self.cam1.imgpoints[imnum][:, 0, :]
        pts3d = self.Triangulation(pts1, pts2)
        X = pts3d[:, 0].reshape(board.size[::-1])
        Y = pts3d[:, 1].reshape(board.size[::-1])
        Z = pts3d[:, 2].reshape(board.size[::-1])
        dX = np.sqrt((X[1:, :]-X[:-1, :])**2 + (Y[1:, :] -
                     Y[:-1, :])**2 + (Z[1:, :]-Z[:-1, :])**2)
        dY = np.sqrt((X[:, 1:]-X[:, :-1])**2 + (Y[:, 1:] -
                     Y[:, :-1])**2 + (Z[:, 1:]-Z[:, :-1])**2)
        print('Step X: mean = %1.2e | std = %1.2e' % (np.mean(dX), np.std(dX)))
        print('Step Y: mean = %1.2e | std = %1.2e' % (np.mean(dY), np.std(dY)))

    def SaveParams(self, filename=None):
        if not filename:
            filename = datetime.now().strftime("cam_param_%y%m%d-%H%M%S-%f")
        np.savez(filename, **self.params)
        print("Writing file %s.npz" % filename)

    def LoadParams(self, filename):
        self.params = dict(np.load(filename))

# %% 

def CamerasFromVICFile(filename):
    """
    Parser for VIC3D calibration file

    Parameters
    ----------
    filename : STRING

    Returns
    -------
    rig : CALIBRATION_TOOL.Stereorig object

    """
    fid = open(filename, "r")
    line = fid.readline()
    ninf = 1000
    for i in range(ninf+1):
        line = fid.readline()
        if line.find("intrinsics") > -1:
            break
    if i == ninf:
        raise Exception('Intrinsic not found in %s' % filename)
    # INTRINSIC 
    def readintrinsic(stop = 'intrinsic'):
        p = np.zeros(10)  # [cx, cy, fx, fy, skew, k1, k2, p1, p2, k3]
        for i in range(ninf+1):
            line = fid.readline()
            if line.find("Center (X)") > -1:
                p[0] = float(line.split()[2])
            if line.find("Center (Y)") > -1:
                p[1] = float(line.split()[2])
            if line.find("Focal Length (X)") > -1:
                p[2] = float(line.split()[3])
            if line.find("Focal Length (Y)") > -1:
                p[3] = float(line.split()[3])
            if line.find("Skew") > -1:
                p[4] = float(line.split()[2])
            if line.find("Kappa 1") > -1:
                p[5] = float(line.split()[2])
            if line.find("Kappa 2") > -1:
                p[6] = float(line.split()[2])
            if line.find("Kappa 3") > -1:
                p[7] = float(line.split()[2])
            if line.find(stop) > -1:
                break
        if i == ninf:
            raise Exception('%s not found in %s' % (stop, filename))
        cam0 = Camera(None)
        cam0.params['Intrinsic'] = np.array(
            [[p[2], p[4], p[0]], [0, p[3], p[1]], [0, 0, 1]])
        cam0.params['Distortion'] = p[5:][np.newaxis]
        return cam0
    cam0 = readintrinsic('intrinsic')
    cam1 = readintrinsic('Extrinsics')
    # EXTRINSIC
    t = np.zeros(3)
    r = np.zeros(3)
    rt = 0
    for i in range(ninf+1):
        line = fid.readline()
        if line.find("Angles") > -1:
            rt += 1
            for i in range(3):
                line = fid.readline()
                rep = line.find(':') + 1
                r[i] = float(line[rep:].split()[0])
        if line.find("Distances") > -1:
            rt += 1
            for i in range(3):
                line = fid.readline()
                rep = line.find(':') + 1
                t[i] = float(line[rep:].split()[0])
        if rt > 1:
            break
    if rt < 2:
        raise Exception('Extrinsic not found in %s' % filename)
    rig = StereoRig(cam0, cam1)
    T = np.zeros((4, 4))
    T[:3, :3] = Rotation.from_euler('xyz', r, degrees=True).as_matrix()
    T[:3, -1] = t
    T[-1, -1] = 1
    rig.params['Transformation'] = T
    return rig


def VICResultsReader(filename):
    """
    Read a VIC3D result file and exports 3D positions,

    Parameters
    ----------
    filename : STRING

    Returns
    -------
    XYZ : TYPE
        3D positions (triangluation).
    xy : TYPE
        coordinates in the camera 0
    qr : TYPE
        left-right stereo matching field

    """
    import glob
    import csv
    csv_files = sorted(glob.glob(filename))
    all_XYZ = []
    all_xy = []
    all_uv = []
    all_qr = []
    all_sigma = []
    for file in csv_files:
        with open(file, 'r', newline='') as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                try:
                    # Extract and convert values in float
                    sigma = float(row["sigma"])
                    X = float(row["X"]) + float(row["U"])
                    Y = float(row["Y"]) + float(row["V"])
                    Z = float(row["Z"]) + float(row["W"])
                    x = float(row["x"])
                    y = float(row["y"])
                    u = float(row["u"])
                    v = float(row["v"])
                    q = float(row["q"])
                    r = float(row["r"])
                except (ValueError, KeyError):
                    # If impossible ou empty column, ignore
                    continue
                all_XYZ.append([X, Y, Z])
                all_xy.append([x, y])
                all_uv.append([u, v])
                all_qr.append([q, r])
                all_sigma.append([sigma])
    # Conversion of list to numpy arrays
    XYZ = np.array(all_XYZ)
    xy = np.array(all_xy)
    qr = np.array(all_qr)
    uv = np.array(all_uv)
    ids, _ = np.where(np.array(all_sigma) > 0)
    XYZ = XYZ[ids, :]
    xy = xy[ids, :]
    qr = qr[ids, :]
    uv = uv[ids, :]
    return XYZ, xy+uv, xy+qr, ids


def CalibrateExtrinsicFrom3DPointsNsys(sys, Xw, uv0, uv1):
    """
    Calibrate extrinsic from 3D points seen in multiple 2-camera systems
    The points must be seen by the 2 camera of each systems, but they
    can be different from one system to the other

    Modify the rig.params['Rotation'] and rig.params['Translation']
    of each systems

    Parameters
    ----------
    sys : PYTHON LIST
        list of CALIBRATION_TOOL.Stereorig
    Xw : PYTHON LIST
        list of numpy array of the coordinates of the points in the new CSYS
    uv0 : PYTHON LIST
        list of coordinates of the projection of the points in the left cam
    uv1 : PYTHON LIST
        list of coordinates of the projection of the points in the right cam

    """
    X = []
    for i in range(len(sys)):
        X += [sys[i].Triangulation(uv0[i], uv1[i])]
    X = np.vstack(X)
    Xw = np.vstack(Xw)
    t = (np.mean(Xw, axis=0) - np.mean(X, axis=0))[:, np.newaxis]
    r = np.zeros((3, 1))

    def residual(p):
        R, _ = cv2.Rodrigues(p[:3, None])
        t = p[3:][:, np.newaxis]
        return (X.T - R@Xw.T - t).ravel()
    params = least_squares(residual, np.append(r, t))
    r = params.x[:3][:, np.newaxis]
    t = params.x[3:][:, np.newaxis]
    for i in range(len(sys)):
        rnew, tnew = ComposeTransformation(
            sys[i].params['Rotation'], sys[i].params['Translation'], r, t)
        sys[i].params['Rotation'] = rnew
        sys[i].params['Translation'] = tnew
