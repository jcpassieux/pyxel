# pyxel
>**py**thon library for e**x**perimental mechanics using finite **el**ements

**pyxel** is an open-source Finite Element (FE) Digital Image Correlation (DIC) library for experimental mechanics application. It is freely available for research and teaching. It is based on `numpy`, `scipy` and `matplotlib`

<p align="center">
  <img src="https://github.com/jcpassieux/pyxel/blob/master/pyxel.png" width="150" title="hover text">
</p>

In its present form, it is restricted to 2D-DIC. Stereo (SDIC) and Digital Volume Correlation (DVC) will be updated later. 
The gray level conservation problem is written in three-dimensional space. It relies on camera models (which must be calibrated) and a dedicated quadrature rule in the 3d space. Considering only 2D-DIC and front-parallel camera settings, the implemented camera model is a simplified pinhole model (including only 4 parameters: 2 translations, 1 rotation and the focal length only). More complex camera models (including distorsions) could easily be implemented within this framework (next update?). The library natively includes linear triangles and quadrilateral elements, but other element types could be added very easily (again?). The library also include de VTK library such that the measurements can be post-processed in Paraview.

1. SCRIPT FILE
    - pyxel is a library. For each testcase, a script file must be written.
    - the testcases (images, meshes...) are stored in the `./data` folder.
    - an sample script named `dic_composite.py` is provided to understand the usage of the library.

2. ABOUT MESHES
    - a mesh is entierly defined by two variables:
        (1) a python dictionnary for the elements (the key is the element number and the value is an
        integer numpy array of size N+1 (N being the number of nodes of this element). The first value
        of this array is the element type (according to gmsh numbering) and the remainder are the node numbers 
        example:
        ```python
        e[num]=np.array([type_el,n0,n1,n2,n3])
        ```
        (2) a numpy array n for the node coordinates
        ```python
        example: n=np.array([[x0,y0],[x1,y1],...])
        ```
    - There is an home made mesher for rectangular and parallelipedic domains.
        give size and number of elements in each direction (see examples).
    - a (not robust) parser for GMSH and Abaqus meshes is embeded in the library. 
        But prefer using existing open-source python parsers. 

3. USING THE LIBRARY
    - Open the mesh: 
      ```python
      m=px.ReadMeshGMSH('data/mesh.msh')
      ```
    - Open the image:
      ```python
      f=px.Image('data/image.tif').Load()
      ```
    - Connectivity, quadrature, Interpolation:
      ```python
      m.Connectivity()
      m.DICIntegration(cam)
      ```      
    - Compute DIC approx. Hessian H and right hand side b:
      ```python
      dic=px.DICEngine()
      H=dic.ComputeLHS(f,m,cam)
      [b,res]=dic.ComputeRHS(g,m,cam,U)
      ```
4. OUTPUT FILES
    - It is possible to post-process the results using matplotlib.
       	but a more convenient way is to use Paraview www.paraview.org 
    - The output files are written in `./vtk` directory 
    - `*.vtu` files are generated, but it is also possible to generate 
      `*.pvd` files for use in paraview.

5. TERM OF USE. 
    This program is a free software: you can redistribute it/or modify it. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.


# References

Jean-Emmanuel Pierré, Jean-Charles Passieux, Jean-Noël Périé. **Finite Element Stereo Digital Image Correlation: Framework and Mechanical Regularization**. *Experimental Mechanics*, Society for Experimental Mechanics, p.443-456, 57, 2017

Jean-Emmanuel Pierré, Jean-Charles Passieux, Jean-Noël Périé, Florian Bugarin, Laurent Robert. **Unstructured finite element-based digital image correlation with enhanced management of quadrature and lens distortions**. *Optics and Lasers in Engineering*, Elsevier, 44-53, 77, 2016. 
