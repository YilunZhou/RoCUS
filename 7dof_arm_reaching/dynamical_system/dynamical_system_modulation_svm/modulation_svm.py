import math, sys
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVR, SVC
from numpy import random as np_random
import learn_gamma_fn
from modulation_utils import *


epsilon = sys.float_info.epsilon

#######################
## Common Functions  ##
#######################
def null_space_bases(n):
    '''construct a set of d-1 basis vectors orthogonal to the given d-dim vector n'''
    d = len(n)
    es = []
    # Random vector
    x = np.random.rand(d)
    # Make it orthogonal to n
    x -= x.dot(n) * n / np.linalg.norm(n)**2
    # normalize it
    x /= np.linalg.norm(x)
    es.append(x)
    # print(np.dot(n,x))
    if np.dot(n,x) > 1e-10:
        raise AssertionError()

    # if 3d make cross product with x
    if d == 3:
        y = np.cross(x,n)
        es.append(y)
        # print(np.dot(n,y), np.dot(x,y))
        if np.dot(n,y) > 1e-10:
            raise AssertionError()
        if np.dot(x,y) > 1e-10:
            raise AssertionError()
    return es


def rand_target_loc(np_random):
    '''
    generate random target location
    '''
    x = np_random.uniform(low=0.05, high=0.5)
    if np_random.randint(0, 2) == 0:
        x = -x
    y = np_random.uniform(low=-0.3, high=0.2)
    z = np_random.uniform(low=0.65, high=1.0)
    return x, z


def linear_controller(x, x_target, max_norm=0.1):
    x_dot = x_target - x
    n = np.linalg.norm(x_dot)
    if n < max_norm:
        return x_dot
    else:
        return x_dot / n * max_norm


########################################################################################################################
## For use with the parametrized gamma functions (circles, rectangles, cross) as well as the RBF defined environments ##
########################################################################################################################

def modulation_single_HBS(x, orig_ds, gamma, verbose = False, ref_adapt = True):
    '''
        Compute modulation matrix for a single obstacle described with a gamma function
        and unique reference point
    '''
    x = np.array(x)
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    normal_vec      = gamma.grad(x)
    reference_point = gamma.center
    gamma_pt        = gamma(x)

    M  = modulation_singleGamma_HBS(x=x, orig_ds=orig_ds, normal_vec=normal_vec, gamma_pt = gamma_pt, reference_point = reference_point, ref_adapt = ref_adapt)
    return M


def modulation_HBS(x, orig_ds, gammas):
    '''
    gammas is a list of k Gamma objects
    '''

    gamma_vals = np.stack([gamma(x) for gamma in gammas])

    if (gamma_vals.shape[0] == 1 and gamma_vals < 1.0) or (gamma_vals.shape[0] > 1 and any(gamma_vals< 1.0) ):
        return np.zeros(orig_ds.shape)

    if (gamma_vals.shape[0] == 1 and gamma_vals > 1e9) or (gamma_vals.shape[0] > 1 and any(gamma_vals > 1e9)):
        return orig_ds

    ms = np.log(gamma_vals - 1 + epsilon)
    logprod = ms.sum()
    bs = np.exp(logprod - ms)
    weights = bs / bs.sum()


    # calculate modulated dynamical systems
    x_dot_mods = []
    for gamma in gammas:
        x_dot_mod = modulation_singleGamma_HBS_multiRef(query_pt=x, orig_ds=orig_ds, gamma_query=gamma(x),
            normal_vec_query=gamma.grad(x), obstacle_reference_points= gamma.center, repulsive_gammaMargin = 0.01)
        x_dot_mods.append(x_dot_mod)
        # The function below does now account for extra modifications on modulated velocity
        # M = modulation_single_HBS(x, gamma)
        # x_dot_mods.append( np.matmul(M, orig_ds.reshape(-1, 1)).flatten() )


    # calculate weighted average of magnitude
    x_dot_mags = np.stack([np.linalg.norm(d) for d in x_dot_mods])
    avg_mag = np.dot(weights, x_dot_mags)

    old_way = True
    if old_way:

        # calculate kappa-space dynamical system and weighted average
        kappas = []
        es = null_space_bases(orig_ds)
        bases = [orig_ds] + es
        R = np.stack(bases).T
        R = R / np.linalg.norm(R, axis=0)

        for x_dot_mod in x_dot_mods:
            n_x_dot_mod = x_dot_mod / np.linalg.norm(x_dot_mod)
            # cob stands for change-of-basis
            n_x_dot_mod_cob = np.matmul(R.T, n_x_dot_mod.reshape(-1, 1)).flatten()
            n_x_dot_mod_cob = n_x_dot_mod_cob / 1.001
            assert -1-1e-5 <= n_x_dot_mod_cob[0] <= 1+1e-5, \
                'n_x_dot_mod_cob[0] = %0.2f?'%n_x_dot_mod_cob[0]
            if n_x_dot_mod_cob[0] > 1:
                acos = np.arccos(n_x_dot_mod_cob[0] - 1e-5)
            elif n_x_dot_mod_cob[0] < -1:
                acos = np.arccos(n_x_dot_mod_cob[0] + 1e-5)
            else:
                acos = np.arccos(n_x_dot_mod_cob[0])
            if np.linalg.norm(n_x_dot_mod_cob[1:]) == 0:
                kappa = acos * n_x_dot_mod_cob[1:] * 0
            else:
                kappa = acos * n_x_dot_mod_cob[1:] / np.linalg.norm(n_x_dot_mod_cob[1:])
            kappas.append(kappa)
        kappas = np.stack(kappas).T

        # matrix-vector multiplication as a weighted sum of columns
        avg_kappa = np.matmul(kappas, weights.reshape(-1, 1)).flatten()

        # map back to task space
        norm = np.linalg.norm(avg_kappa)
        if norm != 0:
            avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa * np.sin(norm) / norm])
        else:
            avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa])
        avg_ds_dir = np.matmul(R, avg_ds_dir.reshape(-1, 1)).flatten()

    else:

        dim = len(x)
        x_dot_mods_normalized = np.zeros((dim, len(gammas)))
        x_dot_mods_np = np.array(x_dot_mods).T

        ind_nonzero = (x_dot_mags>0)
        if np.sum(ind_nonzero):
            x_dot_mods_normalized[:, ind_nonzero] = x_dot_mods_np[:, ind_nonzero]/np.tile(x_dot_mags[ind_nonzero], (dim, 1))
        x_dot_normalized = orig_ds / np.linalg.norm(orig_ds)

        avg_ds_dir = get_directional_weighted_sum(reference_direction=x_dot_normalized,
            directions=x_dot_mods_normalized, weights=weights, total_weight=1)

        x_mod_final = avg_mag*avg_ds_dir.squeeze()

    x_mod_final = avg_mag * avg_ds_dir

    return x_mod_final


def forward_integrate_HBS(x_initial, x_target, gammas, dt, eps, max_N):
    '''
    forward integration of the HBS modulation controller starting from x_initial,
    toward x_target, with obstacles given as a list of gamma functions.
    integration interval is dt, and N integration steps are performed.
    return an (N+1) x d tensor for x trajectory and an N x d tensor for x_dot trajectory.
    '''
    x_traj = []
    x_traj.append(x_initial)
    x_dot_traj = []
    x_cur = x_initial
    for i in range(max_N):
        orig_ds = linear_controller(x_cur, x_target)
        x_dot = modulation_HBS(x_cur, orig_ds, gammas)
        x_dot_traj.append(x_dot)
        x_cur = x_cur + x_dot * dt
        if np.linalg.norm(x_cur - x_target) < eps:
            print("Attractor Reached")
            break
        x_traj.append(x_cur)
    return np.stack(x_traj), np.stack(x_dot_traj)



def forward_integrate_singleGamma_HBS(x_initial, x_target, learned_gamma, dt, eps, max_N):
    '''
    forward integration of the HBS modulation controller starting from x_initial,
    toward x_target, with obstacles given as a list of gamma functions.
    integration interval is dt, and N integration steps are performed.
    return an (N+1) x d tensor for x trajectory and an N x d tensor for x_dot trajectory.
    '''

    # Parse Gamma
    classifier        = learned_gamma['classifier']
    max_dist          = learned_gamma['max_dist']
    reference_points  = learned_gamma['reference_points']
    dim = len(x_target)
    x_traj = []
    x_traj.append(x_initial)
    x_dot_traj = []
    x_cur = x_initial
    # print("Before Integration")
    for i in range(max_N):
        gamma_val  = learn_gamma_fn.get_gamma(x_cur, classifier, max_dist, reference_points, dimension=dim)
        normal_vec = learn_gamma_fn.get_normal_direction(x_cur, classifier, reference_points, max_dist, dimension=dim)        
        orig_ds    = linear_controller(x_cur, x_target)
        x_dot      = modulation_singleGamma_HBS_multiRef(query_pt=x_cur, orig_ds=orig_ds, gamma_query=gamma_val,
                            normal_vec_query=normal_vec.reshape(dim), obstacle_reference_points=reference_points, repulsive_gammaMargin=0.01)
        x_dot      = x_dot/np.linalg.norm(x_dot + epsilon) * 0.10 
        x_dot_traj.append(x_dot)
        x_cur = x_cur + x_dot * dt
        if np.linalg.norm(x_cur - x_target) < eps:
            print("Attractor Reached")
            break
        x_traj.append(x_cur)
    return np.stack(x_traj), np.stack(x_dot_traj)



######################################################################################################################
## For use with non-class defined gamma functions (singleGamma is a single gamma function describing all obstacles) ##
######################################################################################################################
def modulation_singleGamma_HBS(x, orig_ds, normal_vec, gamma_pt, reference_point, ref_adapt = True, tangent_scale_max = 7.5):
    '''
        Compute modulation matrix for a single obstacle described with a gamma function
        and unique reference point
    '''
    x = np.array(x)
    assert len(x.shape) == 1, 'x is not 1-dimensional?'
    d = x.shape[0]
    n = normal_vec

    # Compute the Eigen Bases by adapting the reference direction!
    if ref_adapt:
        E, E_orth = compute_decomposition_matrix(x, normal_vec, reference_point)
    else:
        # Compute the Eigen Bases by NO adaptation of the reference direction!
        es = null_space_bases(n)
        r =  x - reference_point
        bases = [r] + es
        E = np.stack(bases).T
        E = E / np.linalg.norm(E, axis=0)

    invE = np.linalg.inv(E)

    # Compute Diagonal Matrix
    tangent_scaling = 1
    if gamma_pt <=1:
        inv_gamma = 1
    else:
        inv_gamma       = 1 / gamma_pt        
        # Consider TAIL EFFECT!
        tail_angle = np.dot(normal_vec, orig_ds)
        if (tail_angle) < 0:
            # robot moving towards obstacle
            tangent_scaling = max(1, tangent_scale_max - (1-inv_gamma))
        else:
            # robot moving away from  obstacle
            tangent_scaling   = 1.0
            inv_gamma = 0
        
    lambdas = np.stack([1 - inv_gamma] + [tangent_scaling*(1 + inv_gamma)] * (d-1))


    D = np.diag(lambdas)
    M = np.matmul(np.matmul(E, D), invE)


    if gamma_pt > 1e9:
        M =  np.identity(d)

    return M


def modulation_singleGamma_HBS_multiRef(query_pt, orig_ds, gamma_query, normal_vec_query, obstacle_reference_points, repulsive_gammaMargin = 0.01, sticky_surface = False):
    '''
        Computes modulated velocity for an environment described by a single gamma function
        and multiple reference points (describing multiple obstacles)
    '''

    if gamma_query > 1e9:
        return orig_ds

    reference_point = find_closest_reference_point(query_pt, obstacle_reference_points)


    # Added: Move away from center/reference point in case of a collision
    
    if gamma_query < ( 1 + repulsive_gammaMargin):
        repulsive_power =  5
        repulsive_reference_direction = query_pt - reference_point
        x_dot_mod = repulsive_power*(repulsive_reference_direction/np.linalg.norm(repulsive_reference_direction) + orig_ds/np.linalg.norm(orig_ds))
        
        # print('x_dot_mod: ', x_dot_mod)
        # print('reference_point:', reference_point)
    else:
        # Calculate real modulated dynamical system
        M = modulation_singleGamma_HBS(x=query_pt, orig_ds = orig_ds, normal_vec=normal_vec_query, gamma_pt=gamma_query,
            reference_point=reference_point)
        x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()

    # Add: Sticky Surface
    if sticky_surface:
        xd_relative_norm = np.linalg.norm(orig_ds)
        if xd_relative_norm:
            # Limit maximum magnitude
            eigenvalue_magnitude = 1 - 1./abs(gamma_query)**1
            mag = np.linalg.norm(mod_ds)
            mod_ds = mod_ds/mag*xd_relative_norm * eigenvalue_magnitud

    return x_dot_mod


def draw_modulated_svmGamma_HBS(x_target, reference_points, gamma_vals, normal_vecs, nb_obstacles, grid_limits_x, grid_limits_y, grid_size, x_initial, learned_gamma, dt,filename='tmp.png', data = [], sim_type=0):

    fig, ax1 = plt.subplots()
    ax1.set_xlim(grid_limits_x[0], grid_limits_x[1])
    ax1.set_ylim(grid_limits_y[0], grid_limits_y[1])
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title('HBS Modulated DS',fontsize=15)
    X, Y = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    V, U = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            x_query    = np.array([X[i,j], Y[i,j]])
            orig_ds    = linear_controller(x_query, x_target)
            mod_x_dot  = modulation_singleGamma_HBS_multiRef(query_pt=x_query, orig_ds=orig_ds, gamma_query=gamma_vals[i,j],
                                normal_vec_query=normal_vecs[:,i,j], obstacle_reference_points=reference_points, repulsive_gammaMargin=0.01)
            x_dot_norm = mod_x_dot/np.linalg.norm(mod_x_dot + epsilon) * 0.20
            U[i,j]     = x_dot_norm[0]
            V[i,j]     = x_dot_norm[1]

    strm      = ax1.streamplot(X, Y, U, V, density = 3.5, linewidth=0.55, color='k')
    levels    = np.array([0, 1])
    cs0       = ax1.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=2)
    # cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-16, 16, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
    cs        = ax1.contourf(X, Y, gamma_vals, np.arange(-4, 8, 2), cmap=plt.cm.viridis, extend='both', alpha=0.8)
    cbar      = plt.colorbar(cs, orientation='horizontal')
    cbar.add_lines(cs0)

    if len(reference_points) > 1:
        for oo in range(nb_obstacles):
            reference_point = reference_points[oo]
            # print(reference_point)
            ax1.plot([reference_point[0]], [reference_point[1]], 'r+', markersize=12, lw=2)
    else: 
        ax1.plot([reference_points[0]], [reference_points[1]], 'r+',markersize=12, lw=2)       
    plt.gca().set_aspect('equal', adjustable='box')       

    # Integrate trajectories from initial point
    x, x_dot = forward_integrate_singleGamma_HBS(x_initial, x_target, learned_gamma, dt, eps=0.03, max_N = 10000)
    ax1.plot(x.T[0,:], x.T[1,:], 'b.')       

    if data == []:    
        print("Don't plot data")
    else:
        ax1.plot([data[0,:]], [data[1,:]], 'm.')           

    ax1.plot(x_initial[0], x_initial[1], 'gd', markersize=12, lw=2)
    ax1.plot(x_target[0], x_target[1], 'md', markersize=12, lw=2)
    plt.savefig(filename+".png", dpi=300)
    plt.savefig(filename+".pdf", dpi=300)



    # STATUS: WORKS -- NOT USING IT BUT SHOULD ME CLEANED UP IF WE WANT TO COMPARE APPROACHES!!!!
def modulation_multiGamma_HBS_svm(query_pt, orig_ds, learned_gammas, repulsive_gammaMargin = 0.01, sticky_surface = False):
    '''
        Computes modulated velocity for an environment described by a single gamma function
        and multiple reference points (describing multiple obstacles)
    '''

    dim = len(query_pt)
    gamma_vals =[]
    for oo in range(len(learned_gammas)):
        gamma_vals.append(learn_gamma_fn.get_gamma(query_pt.reshape(dim,1), learned_gammas[oo]['classifier'], learned_gammas[oo]['max_dist'], learned_gammas[oo]['reference_point']))

    gamma_vals = np.array(gamma_vals)
    if (len(gamma_vals) == 1 and gamma_vals < 1.0) or (len(gamma_vals) > 1 and any(gamma_vals < 1.0)):
        return np.zeros(orig_ds.shape)

    if (len(gamma_vals) == 1 and gamma_vals > 1e9) or (len(gamma_vals) > 1 and any(gamma_vals > 1e9)):
        return orig_ds

    if len(obstacle_reference_points.shape) > 1:
        reference_point = find_closest_reference_point(query_pt, obstacle_reference_points)
    else:
        reference_point = obstacle_reference_points

    # Add: Move away from center/reference point in case of a collision
    if  0 < gamma_query < ( 1 + repulsive_gammaMargin):
        repulsive_power =  5
        repulsive_factor = 5
        repulsive_gamma = (1 + repulsive_gammaMargin)
        repulsive_speed =  ((repulsive_gamma/gamma_query)**repulsive_power-
                               repulsive_gamma)*repulsive_factor
        repulsive_reference_direction = query_pt - reference_point
        x_dot_mod = repulsive_speed*(1/2)*(repulsive_reference_direction/np.linalg.norm(repulsive_reference_direction) + orig_ds/np.linalg.norm(orig_ds))
        print('SLIDING ON SURFACE!')

    elif gamma_query < 0 :
        x_dot_mod = np.zeros(orig_ds.shape)

    else:
        # Calculate real modulated dynamical system
        M = modulation_singleGamma_HBS(x=query_pt, orig_ds = orig_ds, normal_vec=normal_vec_query, gamma_pt=gamma_query,
            reference_point=reference_point)
        x_dot_mod = np.matmul(M, orig_ds.reshape(-1, 1)).flatten()


    ms = np.log(gamma_vals - 1 + epsilon)
    logprod = ms.sum()
    bs = np.exp(logprod - ms)
    weights = bs / bs.sum()
    weights = weights.T[0]

    # calculate modulated dynamical systems
    x_dot_mods = []
    for oo in range(len(learned_gammas)):
        normal_vec = learn_gamma_fn.get_normal_direction(query_pt.reshape(dim,1), classifier=learned_gammas[oo]['classifier'],reference_point=learned_gammas[oo]['reference_point'],
            max_dist= learned_gammas[oo]['max_dist'], gamma_svm=learned_gammas[oo]['gamma_svm'])
        x_dot_mod = modulation_singleGamma_HBS_multiRef(query_pt=query_pt, orig_ds=orig_ds, gamma_query=gamma_vals[oo][0],
            normal_vec_query=normal_vec.flatten(), obstacle_reference_points = learned_gammas[oo]['reference_point'], repulsive_gammaMargin = 0.01)
        x_dot_mods.append(x_dot_mod)

    # calculate weighted average of magnitude
    x_dot_mags = np.stack([np.linalg.norm(d) for d in x_dot_mods])
    avg_mag = np.dot(weights, x_dot_mags)

    old_way = True
    if old_way:

        # calculate kappa-space dynamical system and weighted average
        kappas = []
        es = null_space_bases(orig_ds)
        bases = [orig_ds] + es
        R = np.stack(bases).T
        R = R / np.linalg.norm(R, axis=0)

        for x_dot_mod in x_dot_mods:
            n_x_dot_mod = x_dot_mod / np.linalg.norm(x_dot_mod)
            # cob stands for change-of-basis
            n_x_dot_mod_cob = np.matmul(R.T, n_x_dot_mod.reshape(-1, 1)).flatten()
            n_x_dot_mod_cob = n_x_dot_mod_cob / 1.001
            assert -1-1e-5 <= n_x_dot_mod_cob[0] <= 1+1e-5, \
                'n_x_dot_mod_cob[0] = %0.2f?'%n_x_dot_mod_cob[0]
            if n_x_dot_mod_cob[0] > 1:
                acos = np.arccos(n_x_dot_mod_cob[0] - 1e-5)
            elif n_x_dot_mod_cob[0] < -1:
                acos = np.arccos(n_x_dot_mod_cob[0] + 1e-5)
            else:
                acos = np.arccos(n_x_dot_mod_cob[0])
            if np.linalg.norm(n_x_dot_mod_cob[1:]) == 0:
                kappa = acos * n_x_dot_mod_cob[1:] * 0
            else:
                kappa = acos * n_x_dot_mod_cob[1:] / np.linalg.norm(n_x_dot_mod_cob[1:])
            kappas.append(kappa)
        kappas = np.stack(kappas).T

        # matrix-vector multiplication as a weighted sum of columns
        avg_kappa = np.matmul(kappas, weights.reshape(-1, 1)).flatten()

        # map back to task space
        norm = np.linalg.norm(avg_kappa)
        if norm != 0:
            avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa * np.sin(norm) / norm])
        else:
            avg_ds_dir = np.concatenate([np.expand_dims(np.cos(norm), 0), avg_kappa])
        avg_ds_dir = np.matmul(R, avg_ds_dir.reshape(-1, 1)).flatten()

    else:

        x_dot_mods_normalized = np.zeros((dim, len(learned_gammas)))
        x_dot_mods_np = np.array(x_dot_mods).T

        ind_nonzero = (x_dot_mags>0)
        if np.sum(ind_nonzero):
            x_dot_mods_normalized[:, ind_nonzero] = x_dot_mods_np[:, ind_nonzero]/np.tile(x_dot_mags[ind_nonzero], (dim, 1))
        x_dot_normalized = orig_ds / np.linalg.norm(orig_ds)

        avg_ds_dir = get_directional_weighted_sum(reference_direction=x_dot_normalized,
            directions=x_dot_mods_normalized, weights=weights, total_weight=1)

        x_mod_final = avg_mag*avg_ds_dir.squeeze()

    x_mod_final = avg_mag * avg_ds_dir