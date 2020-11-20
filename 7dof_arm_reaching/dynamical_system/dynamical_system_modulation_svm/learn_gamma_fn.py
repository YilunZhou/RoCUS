from sklearn import svm
import csv, sys, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from modulation_utils import *

def kernel_first_derivative(query_point, support_vector, gamma_svm, der_wrt):
    """
    Cite: https://github.com/nbfigueroa/SVMGrad/blob/master/matlab/kernel/getKernelFirstDerivative.m

    TODO: this function (or compute_derivatives) has a problem; results are incorrect.
    """
    diff = np.subtract(query_point, support_vector)
    if (der_wrt == 1):
        der_val = gamma_svm * np.exp(-gamma_svm*math.pow(np.linalg.norm(diff),2))*diff
        # der_val = -2 * np.exp(-((gamma_svm * diff).T * diff)) * (gamma_svm * diff)
    else:
        der_val = gamma_svm * np.exp(-gamma_svm*math.pow(np.linalg.norm(diff),2))*np.subtract(support_vector,query_point)
        # der_val = -2 * np.exp(-((gamma_svm * diff).T * diff)) * (gamma_svm * np.subtract(support_vector,query_point))
    return der_val

def compute_derivatives(positions, learned_obstacles, normalize=True):
    """
    Cite: https://github.com/nbfigueroa/SVMGrad/blob/master/matlab/classifier/calculateGammaDerivative.m

    TODO: this function (or kernel_first_derivative) has a problem; results are incorrect.
    """
    normal_vecs     = np.zeros((positions.shape))
    alphas          = learned_obstacles["classifier"].dual_coef_[0]
    bias            = learned_obstacles["classifier"].dual_coef_[1]
    gamma_svm       = learned_obstacles["gamma_svm"]
    support_vectors = learned_obstacles["classifier"].support_vectors_

    for idx in range(len(normal_vecs[0,:])):
        query_point = positions[:,idx]
        normal = np.zeros((query_point.shape))

        for sv_idx in range(len(support_vectors)):
            normal += alphas[sv_idx] * kernel_first_derivative(query_point, support_vectors[sv_idx], gamma_svm=gamma_svm, der_wrt=1)
        normal_vecs[:,idx] = normal

    if normalize:
     mag_normals = np.linalg.norm(normal_vecs, axis=0)
     nonzero_ind = mag_normals>0

     if any(nonzero_ind):
        normal_vecs[:, nonzero_ind] = normal_vecs[:, nonzero_ind] / mag_normals[nonzero_ind]

    return normal_vecs

def get_normal_direction(position, classifier, reference_points, max_dist, gamma_svm=20, normalize=True, delta_dist=1.e-5, dimension=2):
    '''
    Numerical differentiation to of Gamma to get normal direction.
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    '''
    pos_shape = position.shape
    # fairly certain this next line doesn't do anything, but it's in Lukas's code
    positions = position.reshape(dimension, -1)

    # Hacks to modify the normals!!
    if gamma_svm > 0:    
        delta_dist = gamma_svm*delta_dist

    normals = np.zeros((positions.shape))

    for dd in range(dimension):
        pos_low, pos_high = np.copy(positions), np.copy(positions)
        pos_high[dd, :] = pos_high[dd, :] + delta_dist
        pos_low[dd, :] = pos_low[dd, :] - delta_dist
        normals[dd, :] = (get_gamma(pos_high, classifier, max_dist, reference_points, dimension=dimension) - \
                          get_gamma(pos_low, classifier, max_dist, reference_points, dimension=dimension))/(2*delta_dist)

    if normalize:
        mag_normals = np.linalg.norm(normals, axis=0)
        nonzero_ind = mag_normals>0

        if any(nonzero_ind):
            normals[:, nonzero_ind] = normals[:, nonzero_ind] / mag_normals[nonzero_ind]

    return normals

def learn_obstacles_from_data(data_obs, data_free, gamma_svm = 0, C_svm=1000):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    data = np.hstack((data_free, data_obs))
    label = np.hstack(( np.zeros(data_free.shape[1]), np.ones(data_obs.shape[1]) ))

    if gamma_svm == 0:
        n_features = data.T.shape[1]
        gamma_svm = 1 / (n_features * data.T.var())

    classifier = svm.SVC(kernel='rbf', gamma=gamma_svm, C=C_svm).fit(data.T, label)
    print('Number of support vectors / data points')
    print('Free space: ({} / {}) --- Obstacle ({} / {})'.format(
        classifier.n_support_[0], data_free.shape[1],
        classifier.n_support_[1], data_obs.shape[1]))

    if data_obs.shape[0] == 2:
        zero_vec = (0,0)
    else:
        zero_vec = (0,0,0)
    zero_vec = np.zeros(data_obs.shape[0])        
    dist = np.linalg.norm(data_obs-np.tile(zero_vec, (data_obs.shape[1], 1)).T, axis=0)
    max_dist = np.max(dist)

    return classifier, max_dist, gamma_svm, C_svm


def get_gamma(position, classifier, max_dist, reference_points, dimension = 2, truncate = 1):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    pos_shape     = position.shape
    nb_ref_points = len(reference_points)
    svm_bias = classifier.intercept_

    if dimension == 2:
        score = classifier.decision_function(np.c_[position[0].T, position[1].T])
    elif dimension == 3:    
        score = classifier.decision_function(np.c_[position[0].T, position[1].T, position[2].T])
    else:
        error("HIGHER DIMENSIONS NOT SUPPORTED")

    if nb_ref_points == 1:
        reference_point = reference_points
        # This part is doing a version of Eq. 3 of the paper
        dist  = np.linalg.norm(position - np.tile(reference_point, (position.shape[1], 1)).T, axis=0)
    else:
        nb_points = 1
        try: 
            # print(position.shape)
            dist = np.zeros(position.shape[1])

            for i in range(position.shape[1]):
                reference_point = find_closest_reference_point(position[:,i], reference_points)
                diff_vector = position[:,i] - reference_point
                # This part is doing a version of Eq. 3 of the paper
                dist[i]  = np.linalg.norm(diff_vector)
        except:
            reference_point = find_closest_reference_point(position, reference_points)
            dist  = [np.linalg.norm(position - reference_point)]

    # Adapted score to get rid of sink
    truncate = 1
    score_adapt = score
    if truncate:
        bias_scale   = 2.15 # parameter found empirically.. need a better way of doing this
        # bias_scale   = 1.0 # (Gives "faulty" results)
        ind_highvals = score < bias_scale*svm_bias
        # score_adapt[ind_highvals]  = bias_scale*svm_bias

        outer_ref_dist = max_dist*2
        dist = np.clip(dist, max_dist, outer_ref_dist)
        # dist = np.clip(dist, -outer_ref_dist, outer_ref_dist)
        distance_score = (outer_ref_dist-max_dist)/(outer_ref_dist-dist)
        # distance_score = (max_dist)/(outer_ref_dist-dist) # same as above

        # This is not necessary
        # max_float = sys.float_info.max
        # max_float = 1e12
        # gamma = np.zeros(dist.shape)
        # gamma[ind_noninf] = (-score_adapt + 1) * distance_score
        # gamma[~ind_noninf] = max_float

        gamma = (-score_adapt + 1) * distance_score

    else:
        # To visualize the non-truncanted SVM Gamma function
        gamma        = (-score_adapt + 1)


    if len(pos_shape)==1:
        gamma = gamma[0]
    return gamma



def create_obstacles_from_data(data, label, cluster_eps=0.01, cluster_min_samples=10, label_free=0, label_obstacle=1, plot_raw_data=False, gamma_svm=0, c_svm=1000, cluster_labels=[], reference_points=[]):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear

    Parameters:
        data : list of form [[x1 x2 x3 ...], [y1 y2 y3 ...]]
        label : list of form [l1 l2 l3 ...]
        cluster_labels: list of form [c1 c2 c3 ...] 
        cluster_eps : int, default 10. input to DBSCAN.
        cluster_min_samples : int, default 10. input to DBSCAN
        label_free : int, default 0. The label denoting free space.
        label_obstacle : int, default 1. The label denoting obstacles.
        plot_raw_data : boolean, whether to draw raw data

    Returns:
        obstacles : dictionary of form {"classifier": SVM, "max_dist": (float), "obstacle_1": (float, float), "obstacle2": (float, float), ...}

    """
    data_obs = data[:, label==label_obstacle]
    data_free = data[:, label==label_free]

    if plot_raw_data:
        plt.figure(figsize=(6, 6))
        plt.plot(data_free[0, :], data_free[1, :], '.', color='#57B5E5', label='No Collision')
        plt.plot(data_obs[0, :], data_obs[1, :], '.', color='#833939', label='Collision')
        plt.axis('equal')
        plt.title("Raw Data")
        plt.legend()

        plt.xlim([np.min(data[0, :]), np.max(data[0, :])])
        plt.ylim([np.min(data[1, :]), np.max(data[1, :])])
        plt.pause(0.01)
        plt.show()

    learned_obstacles = {}
    classifier, max_dist, gamma_svm, c_svm = learn_obstacles_from_data(data_obs=data_obs, data_free=data_free, 
        gamma_svm=gamma_svm, C_svm=c_svm)
    
    obs_points     = []
    mean_positions = []
    if cluster_labels == [] and reference_points == []:    
        clusters = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(data_obs.T)
        cluster_labels, obs_index = np.unique(clusters.labels_, return_index=True)
        n_obstacles = np.sum(cluster_labels>=0)
        
        print("{} obstacles found with DBSCAN".format(n_obstacles))
        for oo in range(n_obstacles):
            ind_clusters = (clusters.labels_==oo)
            obs_points.append(data_obs[:, ind_clusters])
            mean_position = np.mean(obs_points[-1], axis=1)
            mean_positions.append(mean_position)
    elif reference_points == []:
        n_obstacles = sum(np.unique(cluster_labels) > 0)
        print("{} obstacles given".format(n_obstacles))
        for oo in range(n_obstacles):
            print("obs_id:", oo)
            obs_id = oo + 1
            obstacle_points = data[:,cluster_labels==obs_id]
            mean_position = np.mean(obstacle_points, axis=1)
            mean_positions.append(mean_position)
            print("reference_point:", mean_position)
    else:
        print(reference_points.shape)
        n_obstacles = reference_points.shape[0]
        print("{} obstacles given".format(n_obstacles))
        for oo in range(n_obstacles):
            print("obs_id:", oo)
            obs_id = oo 
            mean_position = reference_points[oo]
            mean_positions.append(mean_position)
            print("reference_point:", mean_position)
    
    learned_obstacles["reference_points"] = mean_positions
    learned_obstacles["classifier"]       = classifier
    learned_obstacles["max_dist"]         = max_dist
    learned_obstacles["gamma_svm"]        = gamma_svm
    learned_obstacles["c_svm"]            = c_svm
    learned_obstacles["n_obstacles"]      = n_obstacles

    return (learned_obstacles)


def create_obstacles_from_data_multi(data, label, cluster_eps = 0.1, cluster_min_samples=10, label_free=0, label_obstacle=1, plot_raw_data=False, gamma_svm=20, c_svm=20.0):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear

    Parameters:
        data : list of form [[x1 x2 x3 ...], [y1 y2 y3 ...]]
        label : list of form [l1 l2 l3 ...]
        cluster_eps : int, default 10. input to DBSCAN.
        cluster_min_samples : int, default 10. input to DBSCAN
        label_free : int, default 0. The label denoting free space.
        label_obstacle : int, default 1. The label denoting obstacles.
        plot_raw_data : boolean, whether to draw raw data

    Returns:
        obstacles : dictionary of form {"classifier": SVM, "max_dist": (float), "obstacle_1": (float, float), "obstacle2": (float, float), ...}

    """
    data_obs = data[:, label==label_obstacle]
    data_free = data[:, label==label_free]

    if plot_raw_data:
        plt.figure(figsize=(6, 6))
        plt.plot(data_free[0, :], data_free[1, :], '.', color='#57B5E5', label='No Collision')
        plt.plot(data_obs[0, :], data_obs[1, :], '.', color='#833939', label='Collision')
        plt.axis('equal')
        plt.title("Raw Data")
        plt.legend()

        plt.xlim([np.min(data[0, :]), np.max(data[0, :])])
        plt.ylim([np.min(data[1, :]), np.max(data[1, :])])
        plt.pause(0.01)
        plt.show()

    # Alternatively - we can use the centers of the generated polygons.
    clusters = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(data_obs.T)
    cluster_labels, obs_index = np.unique(clusters.labels_, return_index=True)
    n_obstacles = np.sum(cluster_labels>=0)
    obs_points = []
    learned_obstacles = {}

    print("{} obstacles found with DBSCAN".format(n_obstacles))

    for oo in range(n_obstacles):
        ind_clusters = (clusters.labels_==oo)
        obs_points.append(data_obs[:, ind_clusters])
        mean_position = np.mean(obs_points[-1], axis=1)
        data_free_temp = np.hstack((data_obs[:, ~ind_clusters], data_free))
        classifier, max_dist, gamma_svm, C_svm = learn_obstacles_from_data(data_obs=obs_points[oo], data_free=data_free_temp, gamma_svm=gamma_svm, C_svm = c_svm)
        
        learned_obstacle_ = {
            "classifier": classifier,
            "max_dist": max_dist,
            "reference_point": mean_position,
            "obstacle_center": mean_position,
            "gamma_svm": gamma_svm,
        }

        learned_obstacles[oo] = learned_obstacle_
        print("Learned gamma for obstacle {}".format(oo))

    return (learned_obstacles)    



def draw_contour_map(classifier, max_dist, reference_points, fig=None, ax=None, show_contour=True, gamma_value=False, normal_vecs = None, show_vecs=True, show_plot = True, grid_limits_x=[0,1], grid_limits_y=[0,1], grid_size=50, data=[]):
    """
    Cite: https://github.com/epfl-lasa/dynamic_obstacle_avoidance_linear
    """
    xx, yy = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))

    if ax is None or fig is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 6)

    if gamma_value:
        predict_score = get_gamma(np.c_[xx.ravel(), yy.ravel()].T, classifier, max_dist, reference_points)
        # predict_score = predict_score - 1 # Subtract 1 to have differentiation boundary at 1 (not sure why this line was necessary)
        plt.title("$\Gamma$-Score")
    else:
        predict_score = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        plt.title("SVM Score")
    predict_score = predict_score.reshape(xx.shape)
    levels = np.array([0, 1])

    cs0 = ax.contour(xx, yy, predict_score, levels, origin='lower', colors='k', linewidths=2)
    if show_contour:

        # cs = ax.contourf(xx, yy, predict_score, np.arange(-16, 16, 2),
        #                  cmap=plt.cm.coolwarm, extend='both', alpha=0.8)

        # Color blind friendly colors
        cs = ax.contourf(xx, yy, predict_score, np.arange(-4, 4, 2),
                         cmap=plt.cm.viridis, extend='both', alpha=0.8)

        cbar = fig.colorbar(cs, orientation='horizontal')
        cbar.add_lines(cs0)

        if show_vecs:
            ax.quiver(xx, yy, normal_vecs[0, :], normal_vecs[1, :])

    else:
        cmap = colors.ListedColormap(['#000000', '#A86161'])
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:, -1] = np.linspace(0.05, 0.7, cmap.N)
        my_cmap = ListedColormap(my_cmap)
        bounds=[-1,0,1]
        norm = colors.BoundaryNorm(bounds, my_cmap.N)

        alphas = 0.5

        cs = ax.contourf(xx, yy, predict_score, origin='lower', cmap=my_cmap, norm=norm)

    if len(reference_points) > 1:
        for oo in range(len(reference_points)):
            reference_point = reference_points[oo]
            # print(reference_point)
            ax.plot([reference_point[0]], [reference_point[1]], 'c+')
    else: 
        ax.plot([reference_points[0]], [reference_points[1]], 'c+')   


    if data == []:    
        print("Don't plot data")
    else:
        ax.plot([data[0,:]], [data[1,:]], 'm.')           

    plt.gca().set_aspect('equal', adjustable='box')

    # if show_plot:
    #     plt.show()

    return (fig, ax)

def read_data (file_location):
    """
    Read data from a CSV

    Parameters:
        file_location : string

    Returns:
        X : numpy array, list of lists, [[x1, x2, x3, ...], [y1, y2, y3, ...]]
        Y : numpy array, list of floats [label1, label 2, ...]
    """
    X = [[],[]]
    Y = []

    with open(file_location, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        header = True
        for row in csvreader:
            # ignore header
            if header:
                header = False
                continue

            X[0].append(int(row[1]))
            X[1].append(int(row[2]))
            Y.append(float(row[0]))

        return (np.array(X), np.array(Y))

def read_data_lasa(file_location):
    """
    Read data from a CSV

    Parameters:
        file_location : string

    Returns:
        X : numpy array, list of lists, [[x1, x2, x3, ...], [y1, y2, y3, ...]]
        Y : numpy array, list of floats [label1, label 2, ...]
    """
    import pandas as pd
    df = pd.read_csv(file_location, sep='[:, ]', engine='python')
    data = np.asarray(df)
    data = np.delete(data, (1,3), axis=1)
    X = data[:, 1:].T
    Y = data[:, 0].T

    return (np.array(X), np.array(Y))



if __name__ == '__main__':
    

    which_data  = 1 # 0: test obstacles, 1: testing data from Lukas' repo
    gamma_type  = 1 # 0:single, 1:multi    

    if which_data:
        # Using data from epfl-lasa github repo
        X, Y = read_data_lasa("dynamical_system_modulation_v2/data/twoObstacles_environment.txt")
        gamma_svm = 20
        c_svm     = 20
        grid_limits = [0, 1]
       
        if not gamma_type:
            # Same SVM for all Gammas (i.e. normals will be the same)
            learned_obstacles = create_obstacles_from_data(data=X, label=Y, plot_raw_data=True, 
                gamma_svm=gamma_svm, c_svm=c_svm)
        else:
            # Independent SVMs for each Gammas (i.e. normals will be different at each grid state)
            learned_obstacles = create_obstacles_from_data_multi(data=X, label=Y, plot_raw_data=True, 
                gamma_svm=gamma_svm, c_svm=c_svm)
         
    else:
        # Using data generated by environments
        # X, Y = read_data("environment_descriptions/tmp.txt")
        X, Y = read_data("dynamical_system_modulation_v2/data/tmp.txt")
        learned_obstacles = create_obstacles_from_data(data=X, label=Y, plot_raw_data=True)
        grid_limits = [0,50]


    # Create Data for plotting    
    xx, yy    = np.meshgrid(np.linspace(grid_limits[0], grid_limits[1], grid_size), np.linspace(grid_limits[0], grid_limits[1], grid_size))
    positions = np.c_[xx.ravel(), yy.ravel()].T
    
    if not gamma_type:

        classifier        = learned_obstacles['classifier']
        max_dist          = learned_obstacles['max_dist']
        reference_point   = learned_obstacles[0]['reference_point']
        gamma_svm         = learned_obstacles['gamma_svm']
        filename          = "./images/ds_tests/svmlearnedGamma_combined.pdf"

        normal_vecs = get_normal_direction(positions, classifier, reference_point, max_dist, gamma_svm=gamma_svm)
        fig,ax      = draw_contour_map(classifier, max_dist, reference_point, gamma_value=True, normal_vecs=normal_vecs, 
            grid_limits_x=grid_limits_x, grid_limits_y=grid_limits_y, grid_size=grid_size)
        fig.savefig(filename, dpi=300)

        # Nice closed form solution, yet computationally inefficient compared to numerical differentiation
        # normal_vecs = compute_derivatives(positions, learned_obstacles)
        # draw_contour_map(classifier, max_dist, gamma_value=True, normal_vecs=normal_vecs, grid_limits=grid_limits, grid_size=grid_size)
    else:
        for oo in range(len(learned_obstacles)):        
            classifier        = learned_obstacles[oo]['classifier']
            max_dist          = learned_obstacles[oo]['max_dist']
            reference_point   = learned_obstacles[oo]['reference_point']
            print("Reference Point:", reference_point)
            gamma_svm         = learned_obstacles[oo]['gamma_svm']
            filename          = './images/ds_tests/svmlearnedGamma_obstacle_{}.pdf'.format(oo)
            print("Gamma SVM:", gamma_svm)

            normal_vecs = get_normal_direction(positions, classifier, reference_point, max_dist, gamma_svm=gamma_svm)
            fig, ax     = draw_contour_map(classifier, max_dist, reference_point, gamma_value=True, normal_vecs=normal_vecs, grid_limits=grid_limits, grid_size=grid_size)
            fig.savefig(filename)
    