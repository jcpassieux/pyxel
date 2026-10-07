# pyxel

> **py**thon library for e**x**perimental mechanics using finite **el**ements

**pyxel** is an open-source Finite Element (FE) Digital Image/Volume Correlation (DIC/DVC) library for experimental mechanics applications. It is freely available for research and teaching.

<p align="center">
  <img src="https://raw.githubusercontent.com/jcpassieux/pyxel/master/pyxel.png" width="150" alt="pyxel logo">
</p>

In its present form, it is restricted to 2D-DIC and 3D-DVC. Stereo-DIC (SDIC) will be added later.

The gray level conservation problem is written in the physical space. It relies on camera models (which must be calibrated) and on a dedicated quadrature rule in the FE mesh space. For front-parallel camera settings, the implemented camera model is a simplified pinhole model, with 4 parameters in 2D (2 translations, 1 rotation and the focal length) and 7 parameters in 3D (focal length, 3 rotations and 3 translations). More complex camera models (including distortions) could easily be implemented within this framework (next update?).

The library natively includes linear and quadratic triangles, quadrilaterals, tetrahedra and hexahedra. Results can be exported in different formats, so that the measurements can be post-processed either with Matplotlib or with Paraview.

<p align="center">
  <img src="https://raw.githubusercontent.com/jcpassieux/pyxel/master/pyxel-figs.png" height="200" alt="pyxel example results">
</p>

## 0. Installation

- Install it from PyPI. The package is named `pyxel-dic`, but it is imported as `pyxel`:

  ```bash
  pip install pyxel-dic
  ```

  ```python
  import pyxel as px
  ```

- Alternatively, clone the git repository and install it in editable mode:

  ```bash
  git clone https://github.com/jcpassieux/pyxel.git
  cd pyxel
  pip install -e .
  ```

- Dependencies (`numpy`, `scipy`, `matplotlib`, `opencv-python`, `scikit-image`, `meshio`, `gmsh`) are installed automatically by `pip`. Python 3 is required.

## 1. Script files

- pyxel is a library: for each test case, a script file must be written.
- A set of tutorials is provided in the `./tests` folder to illustrate the main functionalities of the library.

## 2. About meshes

A Finite Element mesh is required for the displacement measurement. In pyxel, a mesh is entirely defined by two variables:

1. A Python dictionary for the elements. The key is the element type label (gmsh numbering) and the value is a numpy array of size `NE * NN`, where `NN` is the number of nodes of this element type and `NE` the number of elements. Example:

   ```python
   e = dict()
   e[3] = np.array([[n0, n1, n2, n3]])
   ```

2. A numpy array `n` for the node coordinates. Example:

   ```python
   n = np.array([[x0, y0], [x1, y1], ...])
   ```

Notes:

- `pyxel` is able to read and write all the mesh types supported by `meshio`.

## 3. Minimal sample code

To run a simple 2D-DIC analysis:

```python
import numpy as np
import pyxel as px

f = px.Image('img-0.tif').Load()
g = px.Image('img-1.tif').Load()
roi = np.array([[100, 100], [500, 500]])
m, cam = px.MeshFromROI(roi, 50, typel=3)
U, res = px.Correlate(f, g, m, cam)
```

## 4. Output files

Results can be post-processed directly with Matplotlib:

```python
m.PlotContourDispl(U, s=30)
```

A more convenient way (especially in DVC) is to use [Paraview](https://www.paraview.org):

```python
m.VTKSol('vtufile', U)
```

## 5. Terms of use

This program is free software: you can redistribute it and/or modify it. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY.

**pyxel** is distributed under the terms of the [CeCILL](https://cecill.info) license, a French free software license agreement in the spirit of the GNU GPL.

## 6. Dependencies

`numpy`, `scipy`, `matplotlib`, `opencv-python`, `scikit-image`, `meshio`, `gmsh`

## References

Jean-Charles Passieux, Robin Bouclier. **Classic and Inverse Compositional Gauss-Newton in Global DIC**. *International Journal for Numerical Methods in Engineering*, 119(6), p. 453-468, 2019.

Jean-Charles Passieux. **pyxel, an open-source FE-DIC library**. *Zenodo*. [doi:10.5281/zenodo.4654018](http://doi.org/10.5281/zenodo.4654018)

### How to cite

```bibtex
@article{passieux2019classic,
  author  = {Passieux, Jean-Charles and Bouclier, Robin},
  title   = {Classic and Inverse Compositional Gauss-Newton in Global DIC},
  journal = {International Journal for Numerical Methods in Engineering},
  volume  = {119},
  number  = {6},
  pages   = {453--468},
  year    = {2019}
}

@software{passieux_pyxel,
  author = {Passieux, Jean-Charles},
  title  = {pyxel, an open-source FE-DIC library},
  doi    = {10.5281/zenodo.4654018},
  publisher = {Zenodo}
}
```