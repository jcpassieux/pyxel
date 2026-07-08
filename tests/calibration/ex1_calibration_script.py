# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:52:22 2024

@author: passieux
"""
import matplotlib.pyplot as plt
import pyxel as px
import pyxel.calibration as clb

board = clb.Board((11, 7), 'chess', 8)

cam0 = clb.Camera('imgleft')
cam1 = clb.Camera('imgright')

cam0.DetectPoints(board)
cam1.DetectPoints(board)

cam0.Calibrate(board)
cam1.Calibrate(board)

rig = clb.StereoRig(cam0, cam1)
rig.StereoCalibration(board)
rig.PlotParams()

# %% Saving params
cam0.SaveParams('left_cam_params')
cam1.SaveParams('right_cam_params')
rig.SaveParams('rig_params')


# %% and reloading
cam0.LoadParams('left_cam_params.npz')
cam1.LoadParams('right_cam_params.npz')
rig.LoadParams('rig_params.npz')


# %% Improve calibration by removing high reprojection error images
cam0.PlotProjErrors()
cam1.PlotProjErrors()
cam0.RemoveImgHighErrors(1)
cam0.Calibrate(board)
rig = clb.StereoRig(cam0, cam1)
rig.StereoCalibration(board)

# %% Other methods (Board class)
# Plotting the board
board.Plot()
# export as SVG file to print
board.SVGPrint('mire_chess.svg', paper_size=[210, 297])


# %% Other methods (Camera class)

# possibility to define a set of image pre-processing or filtering
opt = dict()
opt['gaussian'] = (3, 3)
cam0.DetectPoints(board, opt)

# plot 0th image of camera
cam0.Plot(0, opt)

# Quick check that the calibration board is found in camera images
res, pts = cam0.DetectPoints_chess(board, 13)
print(res)

# Plot detected points in image 0
cam0.PlotDetectedPoints(board, 0)
# Plot detected points animation
cam0.PlotDetectedPoints(board, wait=200)

# Plot local coordinate system in image 0
cam0.PlotLocalCoordinate(board, 0)
# Plot local coordinate system animation
cam0.PlotLocalCoordinate(board, wait=200)

# Plot calibration parameters
cam0.PlotParams()
cam1.PlotParams()

# %% Other methods (StereoRig class)
# Plot calibration parameters
rig.PlotParams()
# Check the calibration board step size in the 3D space for image 0
rig.CheckBoardStep(board, 0)
# Plot all the calib boards in the global CSYS.
rig.Plot3DBoards()

# %% TRIANGULATION

f = px.Image('')
f.pix = cam0.GetImage(0)
pts0 = f.SelectPoints(6)

f = px.Image('')
f.pix = cam1.GetImage(0)
pts1 = f.SelectPoints(6)

ptsW = rig.Triangulation(pts0, pts1)

ax = plt.figure().add_subplot(projection='3d')
plt.plot(ptsW[:, 0], ptsW[:, 1], ptsW[:, 2], 'ko-')