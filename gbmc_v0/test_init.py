import pickle as pkl
import numpy as np
import os
import sys
import ovito.io as oio;
import ovito.modifiers as ovm;
from ovito.io import import_file, export_file
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from pyhull import qdelaunay

def Circum_O_R(vertex_pos, tol):
    """
    Function finds the center and the radius of the circumsphere of the every tetrahedron. 

cParameters
    -----------------
    vertex_pos :
        The position of vertices of a tetrahedron
    tol :
        Tolerance defined  to identify co-planar tetrahedrons
    
    Returns
    ----------
    circum_center :
        The center of the circum-sphere
    circum_rad :
        The radius of the circum-sphere
    """
    # this is the second method just to check
    # ones_col = np.ones((4,1))
    # a = np.linalg.det(np.append(vertex_pos, ones_col,1))
    # A = vertex_pos[0,:]
    # B = vertex_pos[1,:]
    # C = vertex_pos[2,:]
    # D = vertex_pos[3,:]
    # sq_p1 = np.dot(A,A)
    # sq_p2 = np.dot(B,B)
    # sq_p3 = np.dot(C,C)
    # sq_p4 = np.dot(D,D)

    # dx = np.array([[sq_p1, A[1], A[2], 1], [sq_p2, B[1], B[2], 1],
    #  [sq_p3, C[1], C[2], 1], [sq_p4, D[1], D[2], 1]])
    # Dx = np.linalg.det(dx)

    # dy = np.array([[sq_p1, A[0], A[2], 1], [sq_p2, B[0], B[2], 1],
    #  [sq_p3, C[0], C[2], 1], [sq_p4, D[0], D[2], 1]])
    # Dy = -np.linalg.det(dy)

    # dz = np.array([[sq_p1, A[0], A[1], 1], [sq_p2, B[0], B[1], 1],
    #  [sq_p3, C[0], C[1], 1], [sq_p4, D[0], D[1], 1]])
    # Dz = np.linalg.det(dz)

    # center = [Dx/(2*a), Dy/(2*a), Dz/(2*a)]
    # return center

    # """
    # This function finds thecenter and the radius of the circumsphere of the every tetrahedron.
    # We used "Miroslav Fiedler's method" which is based on  "Cayley-Menger matrix".
    # Ref: https://www.cambridge.org/core/books/matrices-and-graphs-in-geometry/B225AA8C5EA96EAD4D0280173930FE03
    # ------------------------------------
    # Parameter:
    #     vertex_pos: The position of a atoms on vertex.  The dimension of input is (4,3)
    # -------------------------------------
    # Return:
    #     circum_center: The center of the circumsphere
    #     circum_rad: The radius of the circumsphere
    # Test case based on this website: ADD IT TO THE TEST CASE???
    # https://mathematica.stackexchange.com/questions/33846/insphere-for-irregular-tetrahedron/89090#89090
    # a = np.array([[0,0,0],[1,0,0],[2,1,0],[1,1.5,1]])
    # answer:array([ 0.5  ,  1.5  , -0.625]), 1.7001838135919305
    # """
    dis_ij = pdist(vertex_pos, 'euclidean')
    sq_12, sq_13, sq_14, sq_23, sq_24, sq_34 = np.power(dis_ij, 2)

    MatrixC = np.array([[0,1,1,1,1], [1,0,sq_12,sq_13, sq_14], [1,sq_12,0,sq_23,sq_24],
        [1,sq_13,sq_23,0,sq_34],[1,sq_14,sq_24,sq_34,0]])

    det_MC = (np.linalg.det(MatrixC));

    if (det_MC < tol):
        return [0,0,0], 0
    else:
        M = -2*np.linalg.inv(MatrixC)
        circum_center = (M[0,1]*vertex_pos[0,:] + M[0,2]*vertex_pos[1,:] + M[0,3]*vertex_pos[2,:] +
            M[0,4]*vertex_pos[3,:] ) / (M[0,1]+M[0,2]+M[0,3]+M[0,4])
        circum_rad = np.sqrt(M[0,0])/2
        return circum_center, circum_rad



def triang_inds(pts_w_imgs, gb1_inds):
    """
    Function finds the indices of atoms making tetrahedrons
    
    Parameters
    -------------
    pts_w_imgs :
        The position of atoms which are inside the main  box and within rCut to the main box (change name!)
    gb1_inds :
        Indices of the GB atoms

    Returns
    ------------
    tri_vertices :
    gb_tri_inds :
        Tetrahedrons including at least one 
    """
    tri_simplices = qdelaunay("i Qt", pts_w_imgs)
    num_tri = int(tri_simplices[0])
    tri_vertices = np.zeros((num_tri,4), dtype='int')
    for ct1 in range(num_tri):
        tri_vertices[ct1,:] = np.array([int(i) for i in str.split(tri_simplices[ct1+1])])

    tarr1 = np.zeros((np.shape(pts_w_imgs)[0],) )
    tarr1[gb1_inds] = 1
    gb_tri_inds = np.where(np.sum(tarr1[tri_vertices], axis=1))[0]

    return tri_vertices, gb_tri_inds


def vv_props(pts_w_imgs, tri_vertices, gb_tri, lat_par):
    num_tri = np.shape(gb_tri)[0]
    cc_coors = np.zeros((num_tri,3))
    cc_rad = np.zeros((num_tri,1))
    tol = 1e-10*(lat_par**3)
    ct1 = 0
    for tri1 in gb_tri:
        simplex = tri_vertices[tri1, :]
        vertex_pos = pts_w_imgs[simplex,:]
        [cc, cr] = Circum_O_R(vertex_pos, tol)
        cc_coors[ct1,:] = cc; cc_rad[ct1,:] = cr
        ct1 = ct1 + 1

    return cc_coors, cc_rad
