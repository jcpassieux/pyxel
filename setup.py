from setuptools import setup, find_packages
 
setup(
	name="pyxel",
	version="2.0",
	packages=find_packages(), # permet de récupérer tout les fichiers 
	description="pyxel is an open-source Finite Element (FE) \
        Digital Image Correlation (DIC) library for experimental\
        mechanics application ",
	url="https://github.com/jcpassieux/pyxel",
	author="JC Passieux",
	license="CeCiLL",
    install_requires=[
          'meshio', 'gmsh'
      ],
	python_requires=">=3"
	)