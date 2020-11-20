import math, sys
from matplotlib import pyplot as plt
import numpy as np
from numpy import random

# sys.path.append("dynamical_system_modulation_svm/")
import learn_gamma_fn
from modulation_utils import *
import modulation_svm
import pickle

# 3D Plotting
from mpl_toolkits import mplot3d

epsilon = sys.float_info.epsilon

#######################
## Common Functions  ##
#######################
def rand_target_loc(sim_type = 'gaz'):
    '''
    generate random target location
    '''
    if sim_type == 'pyb':
    # ORIGINAL UNIFORM DISTRIBUTION (PYBULLET)
        x = np.random.uniform(low=0.05, high=0.5)
        if np.random.randint(0, 2) == 0:
            x = -x
        y = np.random.uniform(low=-0.3, high=0.2)

    else:    
        # MODIFIED TO ACCOUNT FOR CHANGE IN REFERENCE FRAME (Gazebo)
        y = np.random.uniform(low=0.05, high=0.45)
        if np.random.randint(0, 2) == 0:
            y = -y
        x = np.random.uniform(low = 0.3, high = 0.8)        

    # z = np.random.uniform(low=0.65,  high = 1.0)
    z = np.random.uniform(low=0.65,  high = 0.975)
    return x, y, z

########################################################
##     Tests to run different modulation scenarios    ##
########################################################

def plot_box(ax, center, size, color='C0', alpha=0.2):
    xc, yc, zc = center
    xs, ys, zs = size
    xl, xh = xc - xs / 2, xc + xs / 2
    yl, yh = yc - ys / 2, yc + ys / 2
    zl, zh = zc - zs / 2, zc + zs / 2

    xx, yy = np.meshgrid([xl, xh], [yl, yh])
    ax.plot_surface(xx, yy, np.array([zl, zl, zl, zl]).reshape(2, 2), color=color, alpha=alpha)
    ax.plot_surface(xx, yy, np.array([zh, zh, zh, zh]).reshape(2, 2), color=color, alpha=alpha)

    xx, zz = np.meshgrid([xl, xh], [zl, zh])
    ax.plot_surface(xx, np.array([yl, yl, yl, yl]).reshape(2, 2), zz, color=color, alpha=alpha)
    ax.plot_surface(xx, np.array([yh, yh, yh, yh]).reshape(2, 2), zz, color=color, alpha=alpha)

    yy, zz = np.meshgrid([yl, yh], [zl, zh])
    ax.plot_surface(np.array([xl, xl, xl, xl]).reshape(2, 2), yy, zz, color=color, alpha=alpha)
    ax.plot_surface(np.array([xh, xh, xh, xh]).reshape(2, 2), yy, zz, color=color, alpha=alpha)


def plot_2D_data(X_FREE, X_OBS, grid_limits_x, grid_limits_y):
    X_OBS1,X_OBS2 = X_OBS #Make this general later
    plt.figure()
    plt.axis([grid_limits_x[0], grid_limits_x[1], grid_limits_y[0], grid_limits_y[1]])
    plt.plot(X_FREE[0,:], X_FREE[1,:],'.', color='#57B5E5', label='No Collision')
    plt.plot(X_OBS1[0,:], X_OBS1[1,:],'.', color='#833939', label='Obstacle 1')            
    plt.plot(X_OBS2[0,:], X_OBS2[1,:],'.', color='#833939', label='Obstacle 2')            
    plt.title('Franka ROCUS Collision Dataset',fontsize=15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_3D_data(X_FREE, X_OBS, grid_limits_x, grid_limits_y, grid_limits_z, filename = "franka_environment_collision_dataset"):
    X_OBS1,X_OBS2 = X_OBS #Make this general later
    print("Plotting 3D Data")
    fig = plt.figure()  
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_FREE[0,:],X_FREE[1,:], X_FREE[2,:],'.', alpha= 0.1, color='#57B5E5', label='No Collision');
    ax.scatter3D(X_OBS1[0,:],X_OBS1[1,:], X_OBS1[2,:],'.', color='#833939', label='No Collision');
    ax.scatter3D(X_OBS2[0,:],X_OBS2[1,:], X_OBS2[2,:],'.', color='#833939', label='No Collision');
    ax.view_init(30, -40)
    ax.set_xlabel('$x_1$',fontsize=15)
    ax.set_ylabel('$x_2$',fontsize=15)
    ax.set_zlabel('$x_3$',fontsize=15)
    plt.title('Franka ROCUS Collision Dataset',fontsize=15)    
    plt.savefig(filename+".png", dpi=300)
    plt.savefig(filename+".pdf", dpi=300)
    plt.show()


def create_franka_dataset_gaz_coord(dimension, grid_size, plot_training_data, with_wall = 1):
    grid_limits_x = [ 0.1, 1.0]
    grid_limits_y = [-0.8, 0.8]
    grid_limits_z = [0.55, 1.3]

    xx, yy, zz    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
    X  = np.c_[xx.ravel(), yy.ravel(), zz.ravel()].T
    Y = np.ones(X.shape[1]).T
    C = np.zeros(X.shape[1]).T

    for i in range(X.shape[1]):
        # The table Top
        if (X[2,i] < 0.625):
            Y[i] = 0  
            C[i] = 1

    if with_wall and (dimension==3):         
        for i in range(X.shape[1]):
            # Virtual wall to bound the workspace
            if (X[0,i] > 0.95):
                Y[i] = 0  
                C[i] = 1

    for i in range(X.shape[1]):            
        # The vertical wall
        if (X[0,i]>= 0.3):
          if (X[1,i]>=-0.04 and X[1,i]<=0.04): # Adding 2cm no the sides (to account for gripper)
           if (X[2,i] >= 0.635 and X[2,i] <= 0.975):
             Y[i] = 0
             C[i] = 2        
    
    for i in range(X.shape[1]):            
        # The horizontal wall
        if (X[0,i]>= 0.3):
          if (X[1,i]>=-0.45 and X[1,i]<=0.45): 
              if (X[2,i] >= 0.965 and X[2,i] <= 1.085): 
                Y[i] = 0
                C[i] = 2                          
    Y = 1-Y

    if dimension==2:
        print("Getting a 2D Slice along the x-axis")
        x_slice_idx = (X[0,:] == 1.0) # At 90 cm from robot base
        X_slice = X[:,x_slice_idx]
        Y_slice = Y[x_slice_idx]
        C_slice = C[x_slice_idx]
        if plot_training_data:
            print("Plotting 2D Data")
            X_FREE  = X_slice[:,C_slice.T == 0]
            X_OBS1  = X_slice[:,C_slice.T == 1]
            X_OBS2  = X_slice[:,C_slice.T == 2]
            plot_2D_data(X_FREE[1:3,:], (X_OBS1[1:3,:], X_OBS2[1:3,:]), grid_limits_y, grid_limits_z)

        return X_slice[1:3,:], Y_slice, C_slice
    
    else:     
        if plot_training_data:
            X_FREE = X[:,C.T == 0]
            X_OBS1 = X[:,C.T == 1]
            X_OBS2 = X[:,C.T == 2]
            plot_3D_data(X_FREE, (X_OBS1, X_OBS2), grid_limits_x, grid_limits_y, grid_limits_z, filename = "franka_environment_collision_dataset_gaz")
        return X, Y, C


def create_franka_dataset_pyb_coord(dimension, grid_size, plot_training_data, with_wall = 1):
    grid_limits_x = [-0.8, 0.8, grid_size]
    grid_limits_y = [-0.5, 0.4, grid_size]
    grid_limits_z = [0.55, 1.3, grid_size]

    xx, yy, zz    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
    X  = np.c_[xx.ravel(), yy.ravel(), zz.ravel()].T
    Y = np.ones(X.shape[1]).T
    C = np.zeros(X.shape[1]).T

    for i in range(X.shape[1]):
        # The table Top
        if (X[2,i] < 0.625):
            Y[i] = 0  
            C[i] = 1

    if with_wall and (dimension==3):         
        for i in range(X.shape[1]):
            # Virtual wall to bound the workspace
            if (X[1,i] > 0.35):
                Y[i] = 0  
                C[i] = 1

    for i in range(X.shape[1]):            
        # The vertical wall
        if (X[1,i]>= -0.3):
          if (X[0,i]>=-0.04 and X[0,i]<=0.04): # Adding 2cm no the sides (to account for gripper)
           if (X[2,i] >= 0.635 and X[2,i] <= 0.975):
             Y[i] = 0
             C[i] = 2        
    
    for i in range(X.shape[1]):            
        # The horizontal wall
        if (X[1,i]>= -0.3):
          if (X[0,i]>=-0.45 and X[0,i]<=0.45): 
              if (X[2,i] >= 0.965 and X[2,i] <= 1.085): 
                Y[i] = 0
                C[i] = 2                          
    Y = 1-Y

    if dimension==2:
        print("Getting a 2D Slice along the x-axis")
        x_slice_idx = (X[1,:] == 0.4) # At 90 cm from robot base
        X_slice = X[:,x_slice_idx]
        Y_slice = Y[x_slice_idx]
        C_slice = C[x_slice_idx]
        idx = [0,2]
        if plot_training_data:
            print("Plotting 2D Data")
            X_FREE  = X_slice[:,C_slice.T == 0]
            X_OBS1  = X_slice[:,C_slice.T == 1]
            X_OBS2  = X_slice[:,C_slice.T == 2]            
            plot_2D_data(X_FREE[idx,:],(X_OBS1[idx,:], X_OBS2[idx,:]), grid_limits_x, grid_limits_z)

        return X_slice[idx,:], Y_slice, C_slice
    
    else:     
        if plot_training_data:
            X_FREE = X[:,C.T == 0]
            X_OBS1 = X[:,C.T == 1]
            X_OBS2 = X[:,C.T == 2]
            plot_3D_data(X_FREE, X_OBS, grid_limits_x, grid_limits_x, grid_limits_z, filename = "franka_environment_collision_dataset_pyb")
        return X, Y, C

# --- STATUS: WORKING! -- fix the gamma functions --- #
def learn2D_HBS_svmlearnedGammas(x_target, sim_type):

    grid_size      = 50    
    print('Generating Dataset')
    if sim_type == 'gaz':
        X, Y, c_labels = create_franka_dataset_gaz_coord(dimension=2, grid_size=grid_size, plot_training_data=0, with_wall=0)      
    elif sim_type == 'pyb':
        X, Y, c_labels = create_franka_dataset_pyb_coord(dimension=2, grid_size=grid_size, plot_training_data=0, with_wall=0)      
    print('DONE')

    # Same for both reference frames
    gamma_svm      = 20
    c_svm          = 10
    grid_limits_x  = [-0.8, 0.8]
    grid_limits_y  = [0.55, 1.3]
    dt             = 0.03
    x_initial      = np.array([0.0, 1.221])


    # Same SVM for all Gammas (i.e. normals will be the same)
    print('Learning Gamma function')
    learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
    plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)
    print('Done')

    # Create Data for plotting    
    xx, yy    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    positions = np.c_[xx.ravel(), yy.ravel()].T

    # -- Draw Normal Vector Field of Gamma Functions and Modulated DS with integrated trajectories--- #
    classifier        = learned_obstacles['classifier']
    max_dist          = learned_obstacles['max_dist']
    reference_points  = learned_obstacles['reference_points']
    gamma_svm         = learned_obstacles['gamma_svm']
    n_obstacles       = learned_obstacles['n_obstacles']
    filename          = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_2D"
    if sim_type == 'gaz':
        filename          = filename + "_gaz"
    else:
        filename          = filename + "_pyb"

    normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm)
    fig,ax      = learn_gamma_fn.draw_contour_map(classifier, max_dist, reference_points, gamma_value=True, normal_vecs=normal_vecs, 
        grid_limits_x=grid_limits_x, grid_limits_y=grid_limits_y, grid_size=grid_size)
    fig.savefig(filename+".png", dpi=300)
    fig.savefig(filename+".pdf", dpi=300)

    gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_points)
    # normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm)
    gamma_vals  = gamma_vals.reshape(xx.shape)        
    normal_vecs = normal_vecs.reshape(2, xx.shape[0], xx.shape[1])

    filename = filename + "_modDS"

    # MODULATION CONSIDERS ALL REFERENCE POINTS            
    modulation_svm.draw_modulated_svmGamma_HBS(x_target, reference_points, gamma_vals, normal_vecs, n_obstacles, grid_limits_x,
        grid_limits_y, grid_size, x_initial, learned_obstacles, dt, filename)


# # --- STATUS: WORKING! -- fix the gamma functions --- #
def learn3D_HBS_svmlearnedGammas(x_target, x_initial, sim_type, with_wall):
    # Create 3D Dataset        
    grid_size        = 30
    view_normals     = 0
    view_streamlines = 1
    override_refPts  = 1

    print('Generating Dataset')

    if override_refPts:
        if sim_type == 'pyb':
            reference_points  = np.array([[0, -0.122, 0.975], [0.0, 0.39, 0.776], [0.0, -0.122, 0.61]])
        else:
            reference_points  = np.array([[0.478, 0, 0.975], [0.99, 0, 0.776], [0.478, 0, 0.61]])
    else:
        reference_points = []

    if sim_type == 'gaz':
        X, Y, c_labels = create_franka_dataset_gaz_coord(dimension=3, grid_size=grid_size, plot_training_data=0, with_wall= with_wall)      
        grid_limits_x = [ 0.1, 1.0]
        grid_limits_y = [-0.8, 0.8]
        grid_limits_z = [0.55, 1.3]
    elif sim_type == 'pyb':
        X, Y, c_labels = create_franka_dataset_pyb_coord(dimension=3, grid_size=grid_size, plot_training_data=0, with_wall= with_wall)   
        grid_limits_x = [-0.8, 0.8]
        grid_limits_y = [-0.5, 0.4]
        grid_limits_z = [0.55, 1.3]
    print('DONE')

    gamma_svm      = 20
    c_svm          = 20
    dt             = 0.03    

    # Same SVM for all Gammas (i.e. normals will be the same)
    print('Learning Gamma function')
    learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
    plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels, reference_points=reference_points)
    print('Done')    


    # Save model!
    gamma_svm_model = (learned_obstacles, gamma_svm, c_svm)
    if with_wall:
        filename = "./dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS_bounded"
    else:
        filename = "./dynamical_system_modulation_svm/models/gammaSVM_frankaROCUS"

    if sim_type == 'gaz':        
        pickle.dump(gamma_svm_model, open(filename + "_gaz.pkl", 'wb'))
    else: 
        pickle.dump(gamma_svm_model, open(filename + "_pyb.pkl", 'wb'))

    # -- Draw Normal Vector Field of Gamma Functions and Modulated DS --- #
    xx, yy, zz    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
    positions  = np.c_[xx.ravel(), yy.ravel(), zz.ravel()].T

    classifier        = learned_obstacles['classifier']
    max_dist          = learned_obstacles['max_dist']
    reference_points  = learned_obstacles['reference_points']
    gamma_svm         = learned_obstacles['gamma_svm']
    n_obstacles       = learned_obstacles['n_obstacles']
    filename          = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_3D"
    if sim_type == 'gaz':
        filename          = filename + "_gaz"
    else:
        filename          = filename + "_pyb"


    print("Computing Gammas and Normals..")
    gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_points, dimension=3)
    normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm, dimension=3)
    

    if view_normals:
        ########################################################################################
        # ------ Visualize Gammas and Normal Vectors! -- Make self-contained function ------ #
        fig = plt.figure()  
        ax = plt.axes(projection='3d')
        plt.title("$\Gamma$-Score")
        ax.set_xlim3d(grid_limits_x[0], grid_limits_x[1])
        ax.set_ylim3d(grid_limits_y[0], grid_limits_y[1])
        ax.set_zlim3d(grid_limits_z[0], grid_limits_z[1])
        ax.scatter3D(X[0,:],X[1,:], X[2,:],'.', c=gamma_vals, cmap=plt.cm.coolwarm, alpha = 0.10);

        normal_vecs = normal_vecs.reshape(X.shape)
        ax.quiver(X[0,:],X[1,:], X[2,:], normal_vecs[0,:], normal_vecs[1,:], normal_vecs[2,:], length=0.05, normalize=True)
        ax.view_init(30, 160)
        ax.set_xlabel('$x_1$',fontsize=15)
        ax.set_ylabel('$x_2$',fontsize=15)
        ax.set_zlabel('$x_3$',fontsize=15)
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf", dpi=300)
        plt.show()
        print("DONE")
        ########################################################################################

    if view_streamlines:    
        ########################################################################################
        # ------ Visualize Gammas and Vector Field! -- Make self-contained function ------ #
        fig1 = plt.figure()  
        ax1 = plt.axes(projection='3d')
        plt.title('HBS Modulated DS',fontsize=15)
        ax1.set_xlim3d(grid_limits_x[0], grid_limits_x[1])
        ax1.set_ylim3d(grid_limits_y[0], grid_limits_y[1])
        ax1.set_zlim3d(grid_limits_z[0], grid_limits_z[1])
        filename = filename + "_modDS"
        X_OBS       = X[:,Y == 1]

        # GAMMA_SCORE_OBS = gamma_score[Y == 1]    
        # ax1.scatter3D(X_OBS[0,:],X_OBS[1,:], X_OBS[2,:],edgecolor="r", facecolor="gold");

        vertical_wall = [0, 0.1, 0.825], [0.01, 0.8, 0.4]
        horizontal_wall = [0, 0.1, 1.025], [0.7, 0.8, 0.01]
        table_top = [0, 0, 0.6], [1.5, 1, 0.05]
        plot_box(ax1, *vertical_wall, **{'color': 'C1'})
        plot_box(ax1, *horizontal_wall, **{'color': 'C1'})
        plot_box(ax1, *table_top, **{'color': 'C1'})


        # Integrate trajectories from initial point
        repetitions  = 10
        for i in range(repetitions):
            x_target  = rand_target_loc(sim_type)        
            x, x_dot = modulation_svm.forward_integrate_singleGamma_HBS(x_initial, x_target, learned_obstacles, dt=0.05, eps=0.03, max_N = 10000)
            ax1.scatter(x.T[0,:], x.T[1,:], x.T[2,:], edgecolor="b", facecolor="blue")       

        for oo in range(len(reference_points)):
            reference_point = reference_points[oo]
            print("reference point 1: ", reference_point)
            ax1.scatter3D([reference_point[0]], [reference_point[1]], [reference_point[2]], edgecolor="r", facecolor="red")


        ax1.view_init(0, 180)
        ax1.set_xlabel('$x_1$',fontsize=15)
        ax1.set_ylabel('$x_2$',fontsize=15)
        ax1.set_zlabel('$x_3$',fontsize=15)
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf", dpi=300)
        plt.show()
        print("DONE")
        ########################################################################################

if __name__ == '__main__':

    # --- Test to compare modulation implementations --- #
    dataset_type = '2D'  #  '2D': using a 2D slice of the franka environment, '3D': using the full 3D Franka Environment
    sim_type     = 'pyb' # 'gaz'=gazebo, 'pyb' = pybullet
    with_wall    = 0     # to add virtual wall constraining workspace!

    # Using environment/obstacles learned as svm-defined gamma functions
    if dataset_type == '2D':
        x,y,z     = rand_target_loc(sim_type)
        if sim_type == 'gaz':
            x_target  = np.array([y, z])
        else:
            x_target  = np.array([x, z]) 
        learn2D_HBS_svmlearnedGammas(x_target, sim_type)       

    if dataset_type == '3D':
        x_target  = rand_target_loc(sim_type)
        if sim_type == 'gaz':
            x_initial = np.array([0.516, -0.000, 1.221])
        else:
            x_initial = np.array([0, -0.137, 1.173])
        
        # Test gamma function and generate file for robot control
        learn3D_HBS_svmlearnedGammas(x_target, x_initial, sim_type, with_wall)
