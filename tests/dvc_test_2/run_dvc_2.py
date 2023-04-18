import numpy as np 
import matplotlib.pyplot as plt 
import pyxel as px 
import tools 
from sksparse.cholmod import cholesky


"""
   This scripts performs a shift test on an artificial speckle volume. 
   It also shows how to run the DVC algorithm with an externel linear system solver. 
   Here we use the Cholesky solver provided by the scikit-sparse library 
"""

#%% 
""" Generation of the volumes of the reference and deformed configurations """
res  = 80
N    = [400, 200, 200]
rest = [res ,  int(N[1]/N[0]*res), int(N[2]/N[0]*res)  ]

fpix  = tools.generate_perlin_noise_3d(shape = N, res = rest) 
fpix -= np.min(fpix)
fpix /= np.max(fpix) 
fpix  = (fpix*255).astype('uint8')
# Translating the image f by 10 voxels 
gpix  = tools.translate3DImageFourier(fpix, t=[10,0,0]) 

f = px.Volume(None); f.pix = fpix  ; del fpix 
g = px.Volume(None); g.pix = gpix ; del gpix 
# Building interpolation method 
f.BuildInterp()
g.BuildInterp()

print(f.pix.shape)
f.Plot() 


# Defining the region of interest 
# and meshing it with a structured hexahedral mesh 
roi = np.array([[100,50,50],
                 [300,150,150]])

m = px.StructuredMeshHex8(box=roi, lc=20)

cam = px.CameraVol([1, 0, 0, 0, 0, 0, 0])
px.PlotMeshImage3d(f, m, cam)



#%% INIT
m.Connectivity()
m.DVCIntegration()


""" DVC """ 
# Initial guess 
U0 = np.zeros(m.ndof) 
U0[:m.ndof//3] = 5


dvc = px.DVCEngine()
H     = dvc.ComputeLHS( f, m , cam) 
H_LLt = cholesky(H)

import time 
t = time.time()
U = U0.copy() 
for ik in range(30):
    b, res = dvc.ComputeRHS(g, m, cam, U ) 
    dU = H_LLt.solve_A(b)
    U += dU 
    err=np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | s=%2.2f  | dU/U=%1.2e" % (ik+1,np.std(res),err))
    if err<1e-3 :
        break 
print(time.time()-t)
    

plt.figure()
plt.hist(res,bins=150) 
plt.title('Gray level residual histogram')

plt.figure() 
plt.plot(U)
plt.title('Node displacements')
 




