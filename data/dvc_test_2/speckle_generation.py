from skimage.io import imsave
import numpy as np 
 
res = 80
N   = [400, 200, 200]
rest = [res ,  int(N[1]/N[0]*res), int(N[2]/N[0]*res)  ]

f = utils.generate_perlin_noise_3d( shape=N, res=rest) 
f -= np.min(f)
f /= np.max(f) 
f = (f*255).astype('uint8')
g = utils.translate3DImageFourier(f, t=[10,0,0])

imsave('data/f.tiff', f, plugin='tifffile')
imsave('data/g.tiff', g, plugin='tifffile')


