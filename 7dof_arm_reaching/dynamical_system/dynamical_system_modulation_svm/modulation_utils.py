import math, sys
import numpy as np
from scipy.spatial import distance

def find_closest_reference_point(x, obstacle_reference_points):
    # Use reference point that is nearest; i.e. partition the space ala K-means clustering
    distances = []
    for oo in range(len(obstacle_reference_points)):
        d = distance.euclidean(x, obstacle_reference_points[oo])
        distances.append(d)
    min_idx = np.argmin(distances)
    reference_point = obstacle_reference_points[min_idx]    
    return reference_point

def null_space_bases_old(n):
    '''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
    d = len(n)
    es = []
    for i in range(1, d):
        e = [-n[i]]
        for j in range(1, d):
            if j == i:
                e.append(n[0])
            else:
                e.append(0)
        e = np.stack(e)
        es.append(e)
    return es

def compute_decomposition_matrix(x, normal_vec, reference_point):
    ''' Lukas' version of this function which adapts the reference direction NOT normal '''
    dot_margin=0.05
    reference_direction = - (x - reference_point)
    ref_norm = np.linalg.norm(reference_direction)
    if ref_norm>0:
        reference_direction = reference_direction/ref_norm 
    # print(normal_vec.shape)   
    # print(reference_point.shape)   
    
    dot_prod = np.dot(normal_vec, reference_direction)
    if np.abs(dot_prod) < dot_margin:
        # Adapt reference direction to avoid singularities
        # WARNING: full convergence is not given anymore, but impenetrability
        if not np.linalg.norm(normal_vec): # zero
            normal_vec = -reference_direction
        else:
            weight = np.abs(dot_prod)/dot_margin
            dir_norm = np.copysign(1,dot_prod)
            reference_direction = get_directional_weighted_sum(reference_direction=normal_vec,
                directions=np.vstack((reference_direction, dir_norm*normal_vec)).T,
                weights=np.array([weight, (1-weight)]))
    
    E_orth = get_orthogonal_basis(normal_vec, normalize=True)
    E = np.copy((E_orth))
    E[:, 0] = -reference_direction
    
    return E, E_orth


def get_directional_weighted_sum(reference_direction, directions, weights, total_weight=1, normalize=True, normalize_reference=True):
    '''
    Weighted directional mean for inputs vector ]-pi, pi[ with respect to the reference_direction

    # INPUT
    reference_direction: basis direction for the angle-frame
    directions: the directions which the weighted sum is taken from
    weights: used for weighted sum
    total_weight: [<=1] 
    normalize: 

    # OUTPUT 
    
    '''
    # TODO remove obs and position
    ind_nonzero = (weights>0) # non-negative

    reference_direction = np.copy(reference_direction)
    # print(reference_direction)
    # print(directions)
    directions = directions[:, ind_nonzero] 
    weights = weights[ind_nonzero]

    if total_weight<1:
        weights = weights/np.sum(weights) * total_weight

    n_directions = weights.shape[0]
    if (n_directions==1) and total_weight>=1:
        return directions[:, 0]

    dim = np.array(reference_direction).shape[0]

    if normalize_reference:
        norm_refDir = np.linalg.norm(reference_direction)
        if norm_refDir==0: # nonzero
            raise ValueError("Zero norm direction as input")
        reference_direction /= norm_refDir

     # TODO - higher dimensions
    if normalize:
        norm_dir = np.linalg.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:, ind_nonzero] = directions[:, ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)
    # OrthogonalBasisMatrix = null_space_bases(reference_direction)
    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = np.linalg.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        # Numerical error correction
        cos_directions = np.min(np.vstack((cos_directions, np.ones(n_directions))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(n_directions))), axis=0)
        # warnings.warn("Cosinus value out of bound.") 

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))

    direction_dirSpace_weightedSum = np.sum(directions_directionSpace* np.tile(weights, (dim-1, 1)), axis=1)

    norm_directionSpace_weightedSum = np.linalg.norm(direction_dirSpace_weightedSum)

    if norm_directionSpace_weightedSum:
        direction_weightedSum = (OrthogonalBasisMatrix.dot(
                                  np.hstack((np.cos(norm_directionSpace_weightedSum),
                                              np.sin(norm_directionSpace_weightedSum) / norm_directionSpace_weightedSum * direction_dirSpace_weightedSum)) ))
    else:
        direction_weightedSum = OrthogonalBasisMatrix[:,0]

    return direction_weightedSum


def get_orthogonal_basis(vector, normalize=True):
    '''
    Orthonormal basis for a vector
    '''
    if isinstance(vector, list):
        vector = np.array(vector)
    elif not isinstance(vector, np.ndarray):
        raise TypeError("Wrong input type vector")

    if normalize:
        v_norm = np.linalg.norm(vector)
        if v_norm:
            vector = vector / v_norm
        else:
            raise ValueError("Orthogonal basis Matrix not defined for 0-direction vector.")

    dim = vector.shape[0]
    basis_matrix = np.zeros((dim, dim))

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([basis_matrix[1, 0],
                                       -basis_matrix[0, 0]])
    elif dim == 3:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-vector[1], vector[0], 0])
        
        norm_vec2 = np.linalg.norm(basis_matrix[:, 1])
        if norm_vec2:
            basis_matrix[:, 1] = basis_matrix[:, 1] / norm_vec2
        else:
            basis_matrix[:, 1] = [1, 0, 0]
            
        basis_matrix[:, 2] = np.cross(basis_matrix[:, 0], basis_matrix[:, 1])
        
        norm_vec = np.linalg.norm(basis_matrix[:, 2])
        if norm_vec:
            basis_matrix[:, 2] = basis_matrix[:, 2] / norm_vec
        
    elif dim > 3: # TODO: general basis for d>3
        basis_matrix[:, 0] = vector
        for ii in range(1,dim):
            # TODO: higher dimensions
            if vector[ii]: # nonzero
                basis_matrix[:ii, ii] = vector[:ii]
                basis_matrix[ii, ii] = (-np.sum(vector[:ii]**2)/vector[ii])
                basis_matrix[:ii+1, ii] = basis_matrix[:ii+1, ii]/np.linalg.norm(basis_matrix[:ii+1, ii])
            else:
                basis_matrix[ii, ii] = 1
            # basis_matrix[dim-(ii), ii] = -np.dot(vector[:dim-(ii)], vector[:dim-(ii)])
            # basis_matrix[:, ii] = basis_matrix[:, ii]/LA.norm(basis_matrix[:, ii])

        # import pdb; pdb.set_trace() ## DEBUG ##
        # raise ValueError("Not implemented for d>3")
        # warnings.warn("Implement higher dimensionality than d={}".format(dim))
    return basis_matrix