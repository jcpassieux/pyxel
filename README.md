# pyxel
>**py**thon library for e**x**perimental mechanics using finite **el**ements

**pyxel** is an open-source Finite Element (FE) Digital Image/Volume Correlation (DIC/DVC) library for experimental mechanics application. It is freely available for research and teaching.

<p align="center">
  <img src="https://github.com/jcpassieux/pyxel/blob/master/pyxel.png" width="150" title="hover text">
</p>

In its present form, it is restricted to 2D-DIC and 3D-DVC. Stereo (SDIC) will be updated later. 
The gray level conservation problem is written in the physical space. It relies on camera models (which must be calibrated) and on a dedicated quadrature rule in the FE mesh space. Considering front-parallel camera settings, the implemented camera model is a simplified pinhole model (including only 4 parameters 2D: 2 translations, 1 rotation and the focal length and 7 parameters in 3D: focal length, 3 rotations, 3 translations). More complex camera models (including distorsions) could easily be implemented within this framework (next update?). The library natively includes linear and quadratic triangles, quadrilateral, tetraedral, hexaedral elements. The library also exports the results in different format such that the measurements can be post-processed ether with MatPlotLib or using Paraview.

<p align="center">
  <img src="https://github.com/jcpassieux/pyxel/blob/master/pyxel-figs.png" height="200" title="hover text">
</p>

1. SCRIPT FILE
    - pyxel is a library. For each testcase, a script file must be written.
    - a set of sample scripts named `example_#.py` is provided in the `./test` folder
     to understand the main functionnalities of the library.

2. ABOUT MESHES
    - A Finite Element mesh is required for the displacement measurement. In pyxel, a mesh is entierly defined by two variables:<br>
        (1) a python dictionary for the elements (the key is the element type and the value is a numpy array of size NE * NN (NN being the number of nodes of this element type and NE the number of elements). The element type label (according to gmsh numbering). 
        example:
        ```python
	  e = dict()
       e[3] = np.array([[n0, n1, n2, n3]])
        ```
        (2) a numpy array n for the node coordinates, example:
        ```python
        n = np.array([[x0, y0], [x1, y1], ...])
        ```
    - There is an home made mesher for parallelipedic domains, given the approximate elements size in each direction (see examples). We recommand to use external meshers like, for instance, `gmsh`
    - To read/write the meshes, we use the usefull library `meshio`. 

3. MINIMAL SAMPLE CODE
    - to run a simple 2D-DIC analysis:
      ```python
      f = px.Image('img-0.tif').Load()
      g = px.Image('img-1.tif').Load()
      roi = np.array([[ 100,   100], [ 500,  500]])
      m, cam = px.MeshFromROI(roi, 50, typel=3)
      U, res = px.Correlate(f, g, m, cam)
      ```
    - A multiscale initialization is usually required
      ```python
      U = px.MultiscaleInit(f, g, m, cam, scales=[3, 2, 1], l0=30)
      ```
4. OUTPUT FILES
    - It is possible to post-process the results directly using matplotlib.
      ```python
      m.PlotContourDispl(U, s=30)
      ```
    - but a more convenient way (especially in DVC) is to use Paraview www.paraview.org 
      ```python
      m.VTKSol('vtufile', U)
      ```    
5. TERM OF USE. 
    This program is a free software: you can redistribute it/or modify it. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.
    **pyxel** is distributed under the terms of CeCILL which is a french free software license agreement in the spirit of GNU GPL

6. DEPENDANCIES 
    `numpy`, `scipy`, `matplotlib`, `opencv-python`, `scikit-image`, `meshio`, `gmsh`

7. INSTALL
    - It is possible to clone the git repository, or simply install it using PyPI:
      ```python
      pip install pyxel-dic
      ```
# References

Jean-Charles Passieux, Robin Bouclier. **Classic and Inverse Compositional Gauss-Newton in Global DIC**. *International Journal for Numerical Methods in Engineering*, 119(6), p.453-468, 2019.

Jean-Charles Passieux. **pyxel, an open-source FE-DIC library**. *Zenodo*. http://doi.org/10.5281/zenodo.4654018
