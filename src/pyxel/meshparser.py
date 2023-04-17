






""" ************       USELESS 23/3/2022 !!!    *************** """









import numpy as np
from .mesh import Mesh


#%%
def ReadMeshGMSH(fn, dim=2):
    """
    Beta GMSH parser and converter to pyxel.Mesh object.
    
    Parameters
    ----------
        fn : string (filename)
        dim : int (DEFAULT dim=2)
    """
    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("$Nodes") < 0:
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nnodes = int(line)
    nodes = np.zeros((nnodes, 3))
    for jn in range(nnodes):
        sl = mshfid.readline().split()
        nodes[jn] = np.double(sl[1:])
    while line.find("$Elements") < 0:
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    nelems = int(line)
    elems = dict()
    ne = 1
    line = np.int32(mshfid.readline().split())
    while ne < nelems:
        et = line[1]
        if et == 3:  # qua4
            nn = 4
            rep = np.arange(5, 9)
        elif et == 2:  # tri3
            nn = 3
            rep = np.arange(5, 8)
        elif et == 15:  # point
            nn = 1
            rep = [4]
        elif et == 1:  # segment
            nn = 2
            rep = np.arange(5, 7)
        elif et == 8:  # quadratic segment
            nn = 3
            rep = np.arange(5, 8)
        elif et == 9:  # tri6
            nn = 6
            rep = np.arange(5, 11)
        elif et == 16:  # qua8
            nn = 8
            rep = np.arange(5, 13)
        elif et == 10:  # qua9
            nn = 9
            rep = np.arange(5, 14)
        elems[et] = np.empty((0, nn), dtype=int)
        while line[1] == et:
            elems[et] = np.vstack((elems[et], line[rep] - 1))
            try:
                line = np.int32(mshfid.readline().split())
            except:
                break
            ne += 1
    if dim == 2:
        nodes = np.delete(nodes, 2, 1)
    if 15 in elems.keys():
        del elems[15]  # remove points
    if 1 in elems.keys():
        del elems[1]  # remove segments
    if 8 in elems.keys():
        del elems[8]  # remove quad segments
    m = Mesh(elems, nodes, dim)
    return m

def ReadMeshINP(fn):
    """
    Beta ABAQUS INP parser with 2D mesh and converter to pyxel.Mesh object.
    
    Parameters
    ----------
        fn: string (filename)
    
    2D ONLY : any type of S4, E4, S3, E3, S6, E6, S8, E8
    """

    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("*Node") < 0:
        line = mshfid.readline()
        pass
    line = mshfid.readline()
    dim = len(line.split(","))-1
    nodes = np.zeros((0, dim))
    while line.find("*Element") < 0:
        nodes = np.vstack((nodes, np.double(line.split(",")[1:])))
        line = mshfid.readline()
    elems = dict()
    while "*Element" in line:  # Loop on different element types
        print(line[:-1])
        if "S4" in line or "E4" in line:
            et = 3
            nn = 4
            rep = [1, 2, 3, 4]
        elif "S3" in line or "E3" in line:
            et = 2
            nn = 3
            rep = [1, 2, 3]
        elif "S6" in line or "E6" in line:
            et = 9
            nn = 6
            rep = [1, 2, 3, 4, 5, 6]
        elif "S8" in line or "E8" in line:
            et = 16
            nn = 8
            rep = [1, 2, 3, 4, 5, 6, 7, 8]
        elems[et] = np.empty((0, nn), dtype=int)
        while True:
            line = mshfid.readline().split(",")
            try:
                line = np.int32(line)
            except:
                break
            elems[et] = np.vstack((elems[et], line[rep] - 1))
    m = Mesh(elems, nodes)
    return m

def ReadMeshINPwithElset(fn):
    """
    Beta ABAQUS INP parser with 2D mesh and converter to pyxel.Mesh object.
    Exports also the element sets.

    Parameters
    ----------
        fn : string (filename)
    
    2D ONLY : any type of S4, E4, S3, E3, S6, E6, S8, E8
    """
    mshfid = open(fn, "r")
    line = mshfid.readline()
    while line.find("*Node") < 0:
        line = mshfid.readline()
        pass
    nodes = np.zeros((0, 2))
    line = mshfid.readline()
    while line.find("*Element") < 0:
        nodes = np.vstack((nodes, np.double(line.split(",")[1:])))
        line = mshfid.readline()
    # nnodes = nodes.shape[0]
    elems = dict()
    while "*Element" in line:  # Loop on different element types
        print(line[:-1])
        if "S4" in line or "E4" in line:
            et = 3
            nn = 4
            rep = [1, 2, 3, 4]
        elif "S3" in line or "E3" in line:
            et = 2
            nn = 3
            rep = [1, 2, 3]
        elif "S6" in line or "E6" in line:
            et = 9
            nn = 6
            rep = [1, 2, 3, 4, 5, 6]
        elif "S8" in line or "E8" in line:
            et = 16
            nn = 8
            rep = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            print("Unknown Element!")
            print(line)
        elems[et] = np.empty((0, nn), dtype=int)
        while True:
            line = mshfid.readline()
            try:
                line = np.int32(line.split(","))
            except:
                break
            elems[et] = np.vstack((elems[et], line[rep] - 1))
    elset = dict()
    nelset = 0
    while line.find("*") >= 0:
        if line.find("*Elset") >= 0:
            print(line[:-1])
            if line.find("generate") >= 0:
                line = mshfid.readline()
                gen = np.int32(line.split(","))
                elset[nelset] = np.arange(gen[0] - 1, gen[1], gen[2])
                line = mshfid.readline()
            else:
                line = mshfid.readline()
                lineconcat = ""
                while line.find("*") < 0:
                    lineconcat += "," + line
                    line = mshfid.readline()
                if lineconcat[-2] == ",":
                    lineconcat = lineconcat[:-2]
                elset[nelset] = np.int32(lineconcat[1:].split(",")) - 1
            nelset += 1
        elif line.find("*End Part") >= 0:
            break
        else:
            line = mshfid.readline()
            while line.find("*") < 0:
                line = mshfid.readline()
    m = Mesh(elems, nodes)
    return m, elset


def ReadMeshINP3D(fn):
    """BETA"""
    lines = open(fn, "r").readlines()
    k = 0
    while lines[k] != "*Node\r\n":
        k += 1
    k += 1
    nodes = np.zeros((0, 3))
    while lines[k][0:8] != "*Element":
        nodes = np.vstack((nodes, np.fromstring(lines[k], sep=",")[1:]))
        k += 1
    # here lines[k] == '*Element, type=C3D8R\r\n'
    k += 1
    elems = np.zeros((0, 9), dtype="int")
    while lines[k][0:1] != "*":
        elems = np.vstack((elems, np.fromstring(lines[k], sep=",", dtype="int") - 1))
        k += 1
    elems[:, 0] = 5
    m = Mesh(elems, nodes)
    return m

