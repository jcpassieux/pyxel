import numpy as np 
from skimage.io import imsave   
import matplotlib.pyplot as plt 



def generate_perlin_noise_3d(shape, res):
    """
    Generates a 3D speckle pattern 
    Taken from https://pvigier.github.io/2018/11/02/3d-perlin-noise-numpy.html
    """
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    gradients[-1] = gradients[0]
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def translate3DImageFourier(fpix,t):
    """
    Sub-pixel shift of an image using 3D FFT 
    """
    M1 = fpix.shape[0]
    N1 = fpix.shape[1]
    O1 = fpix.shape[2] 
    
    if M1%2 == 1 and N1%2 ==1 and O1%2==1:
        fpixc = np.zeros((M1+1,N1+1,O1+1))
        fpixc[:-1,:-1,:-1] = fpix
    if M1%2 == 1 and N1%2==1 and O1%2==0:
        fpixc = np.zeros((M1+1,N1+1,O1))
        fpixc[:-1,:-1,:] = fpix
    if M1%2 == 1 and N1%2==0 and O1%2==1:
        fpixc = np.zeros((M1+1,N1,O1+1))
        fpixc[:-1,:,:-1] = fpix
    if M1%2 ==1 and N1%2==0 and O1%2==0 : 
        fpixc = np.zeros((M1+1,N1,O1))
        fpixc[:-1,:,:] = fpix
    if M1%2 == 0 and N1%2==1 and O1%2==1:
        fpixc = np.zeros((M1,N1+1,O1+1))
        fpixc[:,:-1,:-1] = fpix
    if M1%2 == 0 and N1%2==0 and O1%2==1:
        fpixc = np.zeros((M1,N1,O1+1))
        fpixc[:,:,:-1] = fpix 
    if M1%2 == 0 and N1%2==1 and O1%2==0:
        fpixc = np.zeros((M1,N1+1,O1))
        fpixc[:,:-1,:] = fpix   
    if M1%2 == 0 and N1%2==0 and O1%2==0:
        fpixc  = fpix
        
    M = fpixc.shape[0]  
    N = fpixc.shape[1]
    O = fpixc.shape[2]
    
    fftp = np.fft.fftshift(np.fft.fftn(fpixc))
    m1d = np.fft.fftshift(np.fft.fftfreq(M,1/M)) # equivalent m1d = np.arange(-M/2,M/2)
    n1d = np.fft.fftshift(np.fft.fftfreq(N,1/N)) # n1d = np.arange(-N/2,N/2)
    o1d = np.fft.fftshift(np.fft.fftfreq(O,1/O)) 
    m,n,o = np.meshgrid(m1d,n1d,o1d,indexing='ij')
    
    gpix = np.real(np.fft.ifftn(  np.fft.ifftshift(np.exp(-1.j*2*np.pi*( t[0]*m/M +t[1]*n/N + t[2]*o/O  ))*fftp )) )
    
    
    if M1%2 == 1 and N1%2 ==1 and O1%2==1:
        return gpix[:-1,:-1,:-1]  
    if M1%2 == 1 and N1%2==1 and O1%2==0:
        return gpix[:-1,:-1,:]  
    if M1%2 == 1 and N1%2==0 and O1%2==1:
        return gpix[:-1,:,:-1]  
    if M1%2 ==1 and N1%2==0 and O1%2==0 : 
        return gpix[:-1,:,:]  
    if M1%2 == 0 and N1%2==1 and O1%2==1:
        return gpix[:,:-1,:-1]  
    if M1%2 == 0 and N1%2==0 and O1%2==1:
        return gpix[:,:,:-1]  
    if M1%2 == 0 and N1%2==1 and O1%2==0:
        return gpix[:,:-1,:] 
    if M1%2 == 0 and N1%2==0 and O1%2==0:
        return gpix  
