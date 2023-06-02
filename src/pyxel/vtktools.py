#!/usr/bin/env python
""" Some tools to produce VTK XML Files  ***
                JchPassieux 2011

DEMO:

nn=4
ne=2
node=np.array([[0,1,0,1],
               [0,0,1,1],
               [0,0,0,0]])
conn=np.array([0,1,2,1,3,2])
offs=np.array([3,6])
typel=np.array([5,5])
# build the unstructured grid
vtk=VTUWriter(nn,ne,node,conn,offs,typel)

# add point and cell data
displ=np.array([[0,0,0,0],
                [0,0,1,1],
                [0,0,0,0]])
vtk.addPointData('num',1,[0,1,2,3])
vtk.addPointData('displ',3,displ)
vtk.addCellData('num',1,[0,1])

# write file
vtk.write('test')

# another file based on the same mesh
vtk.clearData()
vtk.addPointData('temperature',1,[20,21,22,23])
vtk.addCellData('num',1,[1,2])
vtk.write('test2')


# VTR Example
xi=np.linspace(0.7,1.3,8)
yi=np.linspace(0.7,1.3,8)
zi=np.linspace(0.7,1.3,8)
vtk=VTRWriter(xi,yi,zi)
[Yi,Xi,Zi]=np.meshgrid(yi,xi,zi)
v=(Xi-1)**2+(Yi-1)**2+(Zi-1)**2
vtk.addPointData('u',1,v.T.ravel())
[Yi,Xi,Zi]=np.meshgrid(yi[:-1],xi[:-1],zi[:-1])
E=(Xi-1)**2+(Yi-1)**2+(Zi-1)**2
vtk.addCellData('e',1,E.T.ravel())
vtk.VTRWriter('test2.vtr')

"""

import numpy as np
import os

#import pdb
#pdb.set_trace()

class VTKData:
    def __init__(self,name,numb,vals):
        self.name = name
        self.numb = numb
        if numb > 1:
            self.vals=vals.T.ravel() 
        else:
            self.vals=vals

def array2string(array):
    return ' '.join([str(num) for num in array])


class VTUWriter():
    def __init__(self,nn,ne,node,conn,offs,typel):
        self.clearData()
        self.nn=nn
        self.ne=ne
        self.node=node.T.ravel()
        self.conn=conn
        self.offs=offs
        self.typel=typel
    def addPointData(self,varName,numb,data):
        pdnew=VTKData(varName,numb,data)
        self.pd=np.append(self.pd,pdnew)
    def addCellData(self,varName,numb,data):
        cdnew=VTKData(varName,numb,data)        
        self.cd=np.append(self.cd,cdnew)
    def clearData(self):
        self.pd = np.empty(0 , dtype=object)
        self.cd = np.empty(0 , dtype=object)
  
    def write(self,fileName):
        import xml.dom.minidom
        #import xml.dom.ext # python 2.5 and later
        # Document and root element
        doc = xml.dom.minidom.Document()
        root_element = doc.createElementNS("VTK", "VTKFile")
        root_element.setAttribute("type", "UnstructuredGrid")
        root_element.setAttribute("version", "0.1")
        root_element.setAttribute("byte_order", "LittleEndian")
        doc.appendChild(root_element)
    
        # Unstructured grid element
        unstructuredGrid = doc.createElementNS("VTK", "UnstructuredGrid")
        root_element.appendChild(unstructuredGrid)
    
        # Piece 0 (only one)
        piece = doc.createElementNS("VTK", "Piece")
        piece.setAttribute("NumberOfPoints", str(self.nn))
        piece.setAttribute("NumberOfCells", str(self.ne))
        unstructuredGrid.appendChild(piece)
        
        ### Points ####
        points = doc.createElementNS("VTK", "Points")
        piece.appendChild(points)
    
        # Point location data
        point_coords = doc.createElementNS("VTK", "DataArray")
        point_coords.setAttribute("type", "Float32")
        point_coords.setAttribute("format", "ascii")
        point_coords.setAttribute("NumberOfComponents", "3")
        points.appendChild(point_coords)
        point_coords_data = doc.createTextNode(array2string(self.node))
        point_coords.appendChild(point_coords_data)
    
        #### Cells ####
        cells = doc.createElementNS("VTK", "Cells")
        piece.appendChild(cells)
    
        # Cell Connectivity
        cell_connectivity = doc.createElementNS("VTK", "DataArray")
        cell_connectivity.setAttribute("type", "Int32")
        cell_connectivity.setAttribute("Name", "connectivity")
        cell_connectivity.setAttribute("format", "ascii")        
        cells.appendChild(cell_connectivity)
        connectivity = doc.createTextNode(array2string(self.conn))
        cell_connectivity.appendChild(connectivity)
    
        # Cell Offsets
        cell_offsets = doc.createElementNS("VTK", "DataArray")
        cell_offsets.setAttribute("type", "Int32")
        cell_offsets.setAttribute("Name", "offsets")
        cell_offsets.setAttribute("format", "ascii")                
        cells.appendChild(cell_offsets)
        offsets = doc.createTextNode(array2string(self.offs))
        cell_offsets.appendChild(offsets)
    
        # Cell Types
        cell_types = doc.createElementNS("VTK", "DataArray")
        cell_types.setAttribute("type", "UInt8")
        cell_types.setAttribute("Name", "types")
        cell_types.setAttribute("format", "ascii")                
        cells.appendChild(cell_types)
        types = doc.createTextNode(array2string(self.typel))
        cell_types.appendChild(types)
    
        #### Data at Points ####
        point_data = doc.createElementNS("VTK", "PointData")
        piece.appendChild(point_data)
        for ip in range(len(self.pd)):
            # Points Data
            point_data_array = doc.createElementNS("VTK", "DataArray")
            point_data_array.setAttribute("Name", self.pd[ip].name )
            point_data_array.setAttribute("NumberOfComponents", str(self.pd[ip].numb))
            point_data_array.setAttribute("type", "Float32")
            point_data_array.setAttribute("format", "ascii")
            point_data.appendChild(point_data_array)
            point_data_array_Data = doc.createTextNode(array2string(self.pd[ip].vals))
            point_data_array.appendChild(point_data_array_Data)
    
        #### Cell data (dummy) ####
        cell_data = doc.createElementNS("VTK", "CellData")
        piece.appendChild(cell_data)
        for ic in range(len(self.cd)):
            # Cell Data
            cell_data_array = doc.createElementNS("VTK", "DataArray")
            cell_data_array.setAttribute("Name", self.cd[ic].name )
            cell_data_array.setAttribute("NumberOfComponents", str(self.cd[ic].numb))
            cell_data_array.setAttribute("type", "Float32")
            cell_data_array.setAttribute("format", "ascii")
            cell_data.appendChild(cell_data_array)
            #if cd[ic].numb>1:
            #    cell_data_array_Data = doc.createTextNode(array2string(cd[ic].vals.T.ravel()))
            #else:
            cell_data_array_Data = doc.createTextNode(array2string(self.cd[ic].vals))
            cell_data_array.appendChild(cell_data_array_Data)
    
        # Write to file and exit
        outFile = open(fileName+".vtu", 'w')
        # xml.dom.ext.PrettyPrint(doc, file)
        doc.writexml(outFile, newl='\n')
        print("VTK: "+ fileName +".vtu written")
        outFile.close()
        
class VTRWriter():
    def __init__(self,xi,yi,zi):
        self.clearData()
        self.xi=xi
        self.yi=yi
        self.zi=zi
    def addCellData(self,varName,numb,data):
        cdnew=VTKData(varName,numb,data)
        self.cd=np.append(self.cd,cdnew)
    def addPointData(self,varName,numb,data):
        pdnew=VTKData(varName,numb,data)
        self.pd=np.append(self.pd,pdnew)
    def clearData(self):
        self.cd = np.empty(0 , dtype=object)
        self.pd = np.empty(0 , dtype=object)
    def VTRWriter(self,fileName):
        import xml.dom.minidom    
        # Document and root element
        doc = xml.dom.minidom.Document()
        root_element = doc.createElementNS("VTK", "VTKFile")
        root_element.setAttribute("type", "RectilinearGrid")
        root_element.setAttribute("version", "0.1")
        root_element.setAttribute("byte_order", "LittleEndian")
        doc.appendChild(root_element)
    
        # Unstructured grid element
        RectilinearGrid = doc.createElementNS("VTK", "RectilinearGrid")
        extent=np.array([0,len(self.xi)-1,0,len(self.yi)-1,0,len(self.zi)-1])
        RectilinearGrid.setAttribute("WholeExtent",array2string(extent))
        root_element.appendChild(RectilinearGrid)
    
        # Piece 0 (only one)
        piece = doc.createElementNS("VTK", "Piece")
        piece.setAttribute("Extent", array2string(extent))
        RectilinearGrid.appendChild(piece)
    
        ### Points ####
        points = doc.createElementNS("VTK", "Coordinates")
        piece.appendChild(points)
    
        # Point X Coordinates Data
        point_X_coords = doc.createElementNS("VTK", "DataArray")
        point_X_coords.setAttribute("type", "Float32")
        point_X_coords.setAttribute("Name", "X_COORDINATES")
        point_X_coords.setAttribute("NumberOfComponents", "1")
        point_X_coords.setAttribute("format", "ascii") 
        points.appendChild(point_X_coords)
    
        point_X_coords_data = doc.createTextNode(array2string(self.xi))
        point_X_coords.appendChild(point_X_coords_data)
    
        # Point Y Coordinates Data
        point_Y_coords = doc.createElementNS("VTK", "DataArray")
        point_Y_coords.setAttribute("type", "Float32")
        point_Y_coords.setAttribute("Name", "Y_COORDINATES")
        point_Y_coords.setAttribute("NumberOfComponents", "1")
        point_Y_coords.setAttribute("format", "ascii") 
        points.appendChild(point_Y_coords)
    
        point_Y_coords_data = doc.createTextNode(array2string(self.yi))
        point_Y_coords.appendChild(point_Y_coords_data)
    
        # Point Z Coordinates Data
        point_Z_coords = doc.createElementNS("VTK", "DataArray")
        point_Z_coords.setAttribute("type", "Float32")
        point_Z_coords.setAttribute("Name", "Z_COORDINATES")
        point_Z_coords.setAttribute("NumberOfComponents", "1")
        point_Z_coords.setAttribute("format", "ascii") 
        points.appendChild(point_Z_coords)
    
        point_Z_coords_data = doc.createTextNode(array2string(self.zi))
        point_Z_coords.appendChild(point_Z_coords_data)
    
        #### Cell data  ####
        cell_data = doc.createElementNS("VTK", "CellData")
        piece.appendChild(cell_data)
        for ic in range(len(self.cd)):
            # Cell Data
            cell_data_array = doc.createElementNS("VTK", "DataArray")
            cell_data_array.setAttribute("Name", self.cd[ic].name )
            cell_data_array.setAttribute("NumberOfComponents", str(self.cd[ic].numb))
            cell_data_array.setAttribute("type", "Float32")
            cell_data_array.setAttribute("format", "ascii")
            cell_data.appendChild(cell_data_array)
            cell_data_array_Data = doc.createTextNode(array2string(self.cd[ic].vals))
            cell_data_array.appendChild(cell_data_array_Data)

        #### Point data  ####
        point_data = doc.createElementNS("VTK", "PointData")
        piece.appendChild(point_data)
        for ic in range(len(self.pd)):
            # Point Data
            point_data_array = doc.createElementNS("VTK", "DataArray")
            point_data_array.setAttribute("Name", self.pd[ic].name )
            point_data_array.setAttribute("NumberOfComponents", str(self.pd[ic].numb))
            point_data_array.setAttribute("type", "Float32")
            point_data_array.setAttribute("format", "ascii")
            point_data.appendChild(point_data_array)
            point_data_array_Data = doc.createTextNode(array2string(self.pd[ic].vals))
            point_data_array.appendChild(point_data_array_Data)
    
        # Write to file and exit
        outFile = open(fileName, 'w')
        doc.writexml(outFile, newl='\n')
        print("VTK: "+ fileName +".vtr written")
        outFile.close()

class VTIWriter():
    def __init__(self, nx, ny, nz, sx=0, sy=0, sz=0):
        self.clearData()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.sx = sx
        self.sy = sy
        self.sz = sz
    def addCellData(self, varName, numb, data):
        cdnew = VTKData(varName, numb, data)
        self.cd = np.append(self.cd, cdnew)
    def addPointData(self, varName, numb, data):
        pdnew = VTKData(varName, numb, data)
        self.pd = np.append(self.pd, pdnew)
    def clearData(self):
        self.cd = np.empty(0, dtype=object)
        self.pd = np.empty(0, dtype=object)
    def VTIWriter(self, fileName='output'):
        import xml.dom.minidom    
        # Document and root element
        doc = xml.dom.minidom.Document()
        root_element = doc.createElementNS("VTK", "VTKFile")
        root_element.setAttribute("type", "ImageData")
        root_element.setAttribute("version", "0.1")
        root_element.setAttribute("byte_order", "LittleEndian")
        doc.appendChild(root_element)
    
        # ImageData element
        ImageData = doc.createElementNS("VTK", "ImageData")
        extent=np.array([self.sx, self.sx+self.nx, self.sy, self.sy+self.ny, self.sz, self.sz+self.nz])
        ImageData.setAttribute("WholeExtent", array2string(extent))
        ImageData.setAttribute("Origin", "-0.5 -0.5 -0.5")
        ImageData.setAttribute("Spacing", "1 1 1")
        root_element.appendChild(ImageData)
    
        # Piece 0 (only one)
        piece = doc.createElementNS("VTK", "Piece")
        piece.setAttribute("Extent", array2string(extent))
        ImageData.appendChild(piece)
    
        #### Cell data  ####
        cell_data = doc.createElementNS("VTK", "CellData")
        piece.appendChild(cell_data)
        for ic in range(len(self.cd)):
            # Cell Data
            cell_data_array = doc.createElementNS("VTK", "DataArray")
            cell_data_array.setAttribute("Name", self.cd[ic].name )
            cell_data_array.setAttribute("NumberOfComponents", str(self.cd[ic].numb))
            cell_data_array.setAttribute("type", "Float32")
            cell_data_array.setAttribute("format", "ascii")
            cell_data.appendChild(cell_data_array)
            cell_data_array_Data = doc.createTextNode(array2string(self.cd[ic].vals))
            cell_data_array.appendChild(cell_data_array_Data)

        #### Point data  ####
        point_data = doc.createElementNS("VTK", "PointData")
        piece.appendChild(point_data)
        for ic in range(len(self.pd)):
            # Point Data
            point_data_array = doc.createElementNS("VTK", "DataArray")
            point_data_array.setAttribute("Name", self.pd[ic].name )
            point_data_array.setAttribute("NumberOfComponents", str(self.pd[ic].numb))
            point_data_array.setAttribute("type", "Float32")
            point_data_array.setAttribute("format", "ascii")
            point_data.appendChild(point_data_array)
            point_data_array_Data = doc.createTextNode(array2string(self.pd[ic].vals))
            point_data_array.appendChild(point_data_array_Data)
    
        # Write to file and exit
        fileName += '.vti'
        outFile = open(fileName, 'w')
        doc.writexml(outFile, newl='\n')
        print("VTI: "+ fileName +" written")
        outFile.close()

def PVDFile(fileName,ext,npart,nstep):
    """
    Write PVD file
    Usage: writePVD("toto","vtu",npart,nstep) 
    generated file: "toto.pvd" 
    
    VTK files must be named as follows:
    npart=2  and nstep=5  =>  toto_5_2.*  (starts from zero)
    
    Parameters
    ----------
    fileName : STRING
        mesh files without numbers and extension
    ext : STRING
        extension (vtu, vtk, vtr, vti)
    npart : INT
        Number of parts to plot together
    nstep : INT
        Number of time steps.

    """
    rep, fname = os.path.split(fileName)
    import xml.dom.minidom
    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)
    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)    
    for jp in range(npart):
        for js in range(nstep):
            dataSet = pvd.createElementNS("VTK", "DataSet")
            dataSet.setAttribute("timestep", str(js))
            dataSet.setAttribute("group", "")
            dataSet.setAttribute("part", str(jp))
            dataSet.setAttribute("file", fname+"_"+str(jp)+"_"+str(js)+"."+ext)
            collection.appendChild(dataSet)
    outFile = open(fileName+".pvd", 'w')
    pvd.writexml(outFile, newl='\n')
    print("VTK: "+ fileName +".pvd written")
    outFile.close()
