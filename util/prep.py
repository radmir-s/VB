import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatter3d(shape):
    plt.figure().add_subplot(projection='3d').scatter(*shape.T)

def densify(s, voxel=False, xyz=None, css=None):

    x1, y1, z1 = s.min(axis=0) - 1e-5
    x2, y2, z2 = s.max(axis=0) + 1e-5    

    if xyz:
        dimx, dimy, dimz = xyz
        xx = np.linspace(x1, x2, dimx+1)
        yy = np.linspace(y1, y2, dimy+1)
        zz = np.linspace(z1, z2, dimz+1)
    elif css:
        xx = np.arange(x1, x2 + css, css)
        yy = np.arange(y1, y2 + css, css)
        zz = np.arange(z1, z2 + css, css)
        dimx = len(xx) - 1
        dimy = len(yy) - 1
        dimz = len(zz) - 1
    else:
        print('xyz or css parameters need to be passed in.')
        return None


    sx = s[:,0].reshape(1,-1)
    sy = s[:,1].reshape(1,-1)
    sz = s[:,2].reshape(1,-1)

    indx = (xx[:-1].reshape(-1,1) < sx) & (sx < xx[1:].reshape(-1,1))
    indy = (yy[:-1].reshape(-1,1) < sy) & (sy < yy[1:].reshape(-1,1))
    indz = (zz[:-1].reshape(-1,1) < sz) & (sz < zz[1:].reshape(-1,1))

    indxyz = indx.reshape(dimx,1,1,-1) * indy.reshape(1,dimy,1,-1) * indz.reshape(1,1,dimz,-1)
    dens = np.sum(indxyz, axis=3)
    if voxel:
        return dens
    # dens = dens/dens.sum() # voxel

    densecloud = list()
    for ix, iy, iz in np.argwhere(dens):
        cubeind = indxyz[ix, iy, iz]
        cubepoints = s[cubeind]
        center = cubepoints.mean(axis=0)
        densecloud.append([*center, dens[ix, iy, iz]])

    densecloud = np.array(densecloud)

    return densecloud

def readvtk(vtk_path, align=True, rescale=False):
    with open(vtk_path,'r') as file:
        section = 'header'
        points = []
        polygons = []
        for line in file.readlines():

            if (section == 'header') and line.lower().startswith('points'):
                section = 'points'
                pnum = int(line.split()[1])
                continue
            
            elif (section == 'points') and line.lower().startswith('polygons'):
                section = 'polygons'
                continue

            elif (section == 'polygons') and line.lower().startswith('cell_data'):
                break
            
            match section:
                case 'header':
                    continue
                case 'points':
                    p = list(map(float, line.split()))
                    points.append(p)
                case 'polygons':
                    p = list(map(int, line.split()))
                    polygons.append(p)

    s = np.array(points)
    connections = np.array(polygons)

    if align:
        s = s - s.mean(axis=0)

        pca = PCA(3).fit(s)
        rotation = pca.components_.T if np.linalg.det(pca.components_)>0 else -pca.components_.T
        s = s @ rotation
        
        if rescale:
            s /= pca.singular_values_

    return s, connections