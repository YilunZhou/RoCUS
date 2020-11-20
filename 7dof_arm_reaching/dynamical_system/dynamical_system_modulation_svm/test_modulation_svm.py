import math, sys
from matplotlib import pyplot as plt
import numpy as np
from numpy import random

# sys.path.append("environment_generation/")
# from obstacles import GammaCircle2D, GammaRectangle2D, GammaCross2D
# import sample_environment

sys.path.append("dynamical_system_modulation_svm/")
import learn_gamma_fn
from modulation_utils import *
import modulation_svm
# import rbf_2d_env_svm

# 3D Plotting
from mpl_toolkits import mplot3d

epsilon = sys.float_info.epsilon

#######################
## Common Functions  ##
#######################
def rand_target_loc():
    '''
    generate random target location
    '''
    # ORIGINAL UNIFORM DISTRIBUTION
    # x = np_random.uniform(low=0.05, high=0.5)
    # if np_random.randint(0, 2) == 0:
    #     x = -x
    # y = np_random.uniform(low=-0.3, high=0.2)
    # z = np_random.uniform(low=0.65, high=1.0)

    # MODIFIED TO ACCOUNT FOR CHANGE IN REFERENCE FRAME
    y = np.random.uniform(low=0.05, high=0.45)
    if np.random.randint(0, 2) == 0:
        y = -y
    x = np.random.uniform(low = 0.3, high = 0.8)
    z = np.random.uniform(low=0.65,  high = 1.0)
    return x, y, z


def behavior_length(x_traj):
    '''calculate the total length of the trajectory'''
    diffs = x_traj[1:] - x_traj[:-1]
    dists = np.linalg.norm(diffs, axis=1, ord=2)
    return dists.sum()

########################################################
##     Tests to run different modulation scenarios    ##
########################################################

def create_franka_dataset(dimension, grid_size, plot_training_data):
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
        x_slice_idx = (X[0,:] == 1.0)
        X_slice = X[:,x_slice_idx]
        Y_slice = Y[x_slice_idx]
        C_slice = C[x_slice_idx]
        X_FREE  = X_slice[:,C_slice.T == 0]
        X_OBS1  = X_slice[:,C_slice.T == 1]
        X_OBS2  = X_slice[:,C_slice.T == 2]
        if plot_training_data:
            plt.figure()
            plt.axis([grid_limits_y[0], grid_limits_y[1], grid_limits_z[0], grid_limits_z[1]])
            plt.plot(X_FREE[1,:], X_FREE[2,:],'.', color='#57B5E5', label='No Collision')
            plt.plot(X_OBS1[1,:], X_OBS1[2,:],'.', color='#833939', label='Obstacle 1')            
            plt.plot(X_OBS2[1,:], X_OBS2[2,:],'.', color='#833939', label='Obstacle 2')            
            plt.title('Franka ROCUS Collision Dataset',fontsize=15)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        return X_slice[1:3,:], Y_slice, C_slice
    else: 
        
        X_FREE      = X[:,C.T == 0]
        X_OBS1      = X[:,C.T == 1]
        X_OBS2      = X[:,C.T == 2]
        if plot_training_data:
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
            filename = "franka_environment_collision_dataset"
            plt.savefig(filename+".png", dpi=300)
            plt.savefig(filename+".pdf", dpi=300)

            plt.show()
        return X, Y, C


def create_franka_dataset_pyb_coord(grid_size=30, plot_training_data=False):
    lin_x = [-0.8, 0.8, grid_size]
    lin_y = [-0.5, 0.4, grid_size]
    lin_z = [0.55, 1.3, grid_size]

    xx, yy, zz = np.meshgrid(np.linspace(*lin_x), np.linspace(*lin_y), np.linspace(*lin_z))
    X  = np.c_[xx.ravel(), yy.ravel(), zz.ravel()].T  # 3 x N array with each column representing a data point
    Y  = np.ones(X.shape[1]).T
    C  = np.zeros(X.shape[1]).T
    # table top: z < 0.625
    # vertical: -0.03 <= x <= 0.03 and y >= -0.3 and 0.635 <= z <= 0.975
    # horizontal: -0.45 <= x <= 0.45 and y >= -0.3 and 0.965 <= z <= 1.085
    # virtual: y > 0.3
    for i in range(X.shape[1]):
        x, y, z = X[:, i]
        tabletop = (z < 0.625)
        vertical = (-0.03 <= x <= 0.03 and y >= -0.3 and 0.635 <= z <= 0.975)
        horizontal = (-0.45 <= x <= 0.45 and y >= -0.3 and 0.965 <= z <= 1.085)
        virtual = (y > 0.3)
        if tabletop or vertical or horizontal or virtual:
            Y[i] = 0
    Y = 1 - Y
    return X, Y, None

# --- STATUS: WORKING! -- fix the gamma functions --- #
def test2D_HBS_svmlearnedGammas(x_target, gamma_type, which_data):
    # Using 2D data from epfl-lasa dataset
    if (which_data == 0):           
        grid_size   = 50
        X, Y = learn_gamma_fn.read_data_lasa("dynamical_system_modulation_svm/data/twoObstacles_environment.txt")
        gamma_svm = 20
        c_svm     = 20
        grid_limits_x  = [0, 1]
        grid_limits_y   = [0, 1]
        c_labels = []
        dt       = 0.03
        x_initial      = np.array([0.0, 0.65])

    # Using 2D data from irg-frank/table setup
    if (which_data == 1):           
        grid_size      = 50
        X, Y, c_labels = create_franka_dataset(dimension=2, grid_size=grid_size, plot_training_data=0)      
        gamma_svm      = 20
        c_svm          = 10
        grid_limits_x  = [-0.8, 0.8]
        grid_limits_y  = [0.55, 1.3]
        dt             = 0.03
        x_initial      = np.array([0.0, 1.221])

    # Learn Gamma Function/s!
    if not gamma_type:
        # Same SVM for all Gammas (i.e. normals will be the same)
        learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
            plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)
    else:
        # Independent SVMs for each Gammas (i.e. normals will be different at each grid state)
        learned_obstacles = learn_gamma_fn.create_obstacles_from_data_multi(data=X, label=Y,
            plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)

    # Create Data for plotting    
    xx, yy    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size))
    positions = np.c_[xx.ravel(), yy.ravel()].T
    
    # -- Draw Normal Vector Field of Gamma Functions and Modulated DS --- #
    if not gamma_type:
        # This will use the single gamma function formulation (which is not entirely correct due to handling of the reference points)
        classifier        = learned_obstacles['classifier']
        max_dist          = learned_obstacles['max_dist']
        reference_points  = learned_obstacles['reference_points']
        gamma_svm         = learned_obstacles['gamma_svm']
        n_obstacles       = learned_obstacles['n_obstacles']
        filename          = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_2D"

        normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm)
        fig,ax      = learn_gamma_fn.draw_contour_map(classifier, max_dist, reference_points, gamma_value=True, normal_vecs=normal_vecs, 
            grid_limits_x=grid_limits_x, grid_limits_y=grid_limits_y, grid_size=grid_size, data=X[:,Y==1])
        fig.savefig(filename+".png", dpi=300)
        fig.savefig(filename+".pdf", dpi=300)

        gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_points)
        normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm)
        gamma_vals  = gamma_vals.reshape(xx.shape)        
        normal_vecs = normal_vecs.reshape(2, xx.shape[0], xx.shape[1])

        filename = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_modDS_2D"

        # MODULATION CONSIDERS ALL REFERENCE POINTS            
        modulation_svm.draw_modulated_svmGamma_HBS(x_target, reference_points, gamma_vals, normal_vecs, n_obstacles, grid_limits_x,
            grid_limits_y, grid_size, x_initial, learned_obstacles, dt, filename, data=X[:,Y==1])

    else:
        for oo in range(len(learned_obstacles)):        
            classifier        = learned_obstacles[oo]['classifier']
            max_dist          = learned_obstacles[oo]['max_dist']
            reference_point   = learned_obstacles[oo]['reference_point']
            gamma_svm         = learned_obstacles[oo]['gamma_svm']

            filename          = './dynamical_system_modulation_svm/figures/svmlearnedGamma_obstacle_{}_2D.png'.format(oo)

            normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_point, max_dist, gamma_svm=gamma_svm)
            fig, ax     = learn_gamma_fn.draw_contour_map(classifier, max_dist, reference_point, gamma_value=True, normal_vecs=normal_vecs, 
                grid_limits=grid_limits, grid_size=grid_size)
            fig.savefig(filename)

            print("Doing modulation for obstacle {}".format(oo))
            gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_point)

            # This will use the single gamma function formulation (which
            gamma_vals  = gamma_vals.reshape(xx.shape)        
            normal_vecs = normal_vecs.reshape(2, xx.shape[0], xx.shape[1])
            filename = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_obstacle_{}_modDS_2D.png".format(oo)
            modulation_svm.raw_modulated_svmGamma_HBS(x_target, reference_point, gamma_vals, normal_vecs, 1, grid_limits, grid_size, filename)
        
        print("Doing combined modulation of all obstacles")
        filename = "./dynamical_system_modulation_svm/figures/multisvmlearnedGamma_ALLobstacles_modDS_2D.png"
        modulation_svm.draw_modulated_multisvmGamma_HBS(x_target, learned_obstacles, grid_limits, grid_size, filename)

# # --- STATUS: WORKING! -- fix the gamma functions --- #
def test3D_HBS_svmlearnedGammas(x_target, x_initial, gamma_type):
    # Create 3D Dataset        
    grid_size      = 15
    X, Y, c_labels = create_franka_dataset(dimension=3, grid_size=grid_size, plot_training_data=0)      
    gamma_svm      = 20
    c_svm          = 20
    grid_limits_x  = [0.1, 1.0]
    grid_limits_y  = [-0.8, 0.8]
    grid_limits_z  = [0.55, 1.1]
    dt             = 0.03    

    # Learn Gamma Function/s!
    if not gamma_type:
        # Same SVM for all Gammas (i.e. normals will be the same)
        learned_obstacles = learn_gamma_fn.create_obstacles_from_data(data=X, label=Y, 
            plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)
    else:
        # Independent SVMs for each Gammas (i.e. normals will be different at each grid state)
        learned_obstacles = learn_gamma_fn.create_obstacles_from_data_multi(data=X, label=Y,
            plot_raw_data=False,  gamma_svm=gamma_svm, c_svm=c_svm, cluster_labels = c_labels)


    # -- Draw Normal Vector Field of Gamma Functions and Modulated DS --- #
    grid_limits_x = [ 0.1, 1.0]
    grid_limits_y = [-0.8, 0.8]
    grid_limits_z = [0.55, 1.3]

    xx, yy, zz    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
    positions  = np.c_[xx.ravel(), yy.ravel(), zz.ravel()].T

    if not gamma_type:
        # This will use the single gamma function formulation (which is not entirely correct due to handling of the reference points)
        classifier        = learned_obstacles['classifier']
        max_dist          = learned_obstacles['max_dist']
        reference_points  = learned_obstacles['reference_points']
        gamma_svm         = learned_obstacles['gamma_svm']
        n_obstacles       = learned_obstacles['n_obstacles']
        filename          = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_3D"

        print("Computing Gammas and Normals..")
        gamma_vals  = learn_gamma_fn.get_gamma(positions, classifier, max_dist, reference_points, dimension=3)
        normal_vecs = learn_gamma_fn.get_normal_direction(positions, classifier, reference_points, max_dist, gamma_svm=gamma_svm, dimension=3)
        

        ########################################################################################
        # ------ Visualize Gammas and Normal Vectors! -- Make self-contained function ------ #
        fig = plt.figure()  
        ax = plt.axes(projection='3d')
        plt.title("$\Gamma$-Score")
        ax.set_xlim3d(grid_limits_x[0], grid_limits_x[1])
        ax.set_ylim3d(grid_limits_y[0], grid_limits_y[1])
        ax.set_zlim3d(grid_limits_z[0], grid_limits_z[1])
        gamma_score = gamma_vals - 1 # Subtract 1 to have differentiation boundary at 1
        ax.scatter3D(X[0,:],X[1,:], X[2,:],'.', c=gamma_score, cmap=plt.cm.coolwarm, alpha = 0.10);

        normal_vecs = normal_vecs.reshape(X.shape)
        ax.quiver(X[0,:],X[1,:], X[2,:], normal_vecs[0,:], normal_vecs[1,:], normal_vecs[2,:], length=0.05, normalize=True)
        ax.view_init(30, -40)
        ax.set_xlabel('$x_1$',fontsize=15)
        ax.set_ylabel('$x_2$',fontsize=15)
        ax.set_zlabel('$x_3$',fontsize=15)
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf", dpi=300)
        plt.show()
        print("DONE")
        ########################################################################################


        ########################################################################################
        # ------ Visualize Gammas and Vector Field! -- Make self-contained function ------ #
        fig1 = plt.figure()  
        ax1 = plt.axes(projection='3d')
        plt.title('HBS Modulated DS',fontsize=15)
        ax.set_xlim3d(grid_limits_x[0], grid_limits_x[1])
        ax.set_ylim3d(grid_limits_y[0], grid_limits_y[1])
        ax.set_zlim3d(grid_limits_z[0], grid_limits_z[1])
        filename    = "./dynamical_system_modulation_svm/figures/svmlearnedGamma_combined_modDS_3D"

        X_OBS   = X[:,Y == 1]
        # GAMMA_SCORE_OBS = gamma_score[Y == 1]
        ax1.scatter3D(X_OBS[0,:],X_OBS[1,:], X_OBS[2,:],edgecolor="r", facecolor="gold");

        # Add vector field!
        print("Computing the Vector Field.")   
        # NOT NECESSARY!!!!
        # Create data for 3D Visualization of vector fields
        xx, yy, zz   = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), 
            np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
        normal_vecs   = normal_vecs.reshape(3, xx.shape[0], xx.shape[1], xx.shape[2])
        uu, vv, ww    = np.meshgrid(np.linspace(grid_limits_x[0], grid_limits_x[1], grid_size), 
            np.linspace(grid_limits_y[0], grid_limits_y[1], grid_size), np.linspace(grid_limits_z[0], grid_limits_z[1], grid_size))
        gamma_vals  = gamma_vals.reshape(xx.shape)        
        print(gamma_vals.shape)
        normal_vecs = normal_vecs.reshape(3, xx.shape[0], xx.shape[1], xx.shape[2])
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    x_query    = np.array([xx[i,j,k], yy[i,j,k], zz[i,j,k]])
                    orig_ds    = modulation_svm.linear_controller(x_query, x_target)
                    print(gamma_vals[i,j,k])
                    print("orig_ds",orig_ds)
                    mod_x_dot  = modulation_svm.modulation_singleGamma_HBS_multiRef(query_pt=x_query, orig_ds=orig_ds, gamma_query=gamma_vals[i,j,k],
                                        normal_vec_query=normal_vecs[:,i,j,k], obstacle_reference_points=reference_points, repulsive_gammaMargin=0.01)
                    x_dot_norm = mod_x_dot/np.linalg.norm(mod_x_dot + epsilon)*0.05
                    uu[i,j]     = x_dot_norm[0]
                    vv[i,j]     = x_dot_norm[1]
                    ww[i,j]     = x_dot_norm[2]
        
        ax1.quiver(xx,yy, zz, uu, vv, ww, length=0.05, normalize=True, alpha = 0.1)
        ax1.view_init(30, -40)
        print("Done plotting vector field.") 


        # Integrate trajectories from initial point
        repetitions  = 10
        for i in range(repetitions):
            x_target  = rand_target_loc()        
            x, x_dot = modulation_svm.forward_integrate_singleGamma_HBS(x_initial, x_target, learned_obstacles, dt=0.05, eps=0.03, max_N = 10000)
            ax1.scatter(x.T[0,:], x.T[1,:], x.T[2,:], edgecolor="b", facecolor="blue")       


        print("reference_points", reference_points)
        reference_point = reference_points[0]
        print("reference point 1: ", reference_point)
        ax1.scatter3D([reference_point[0]], [reference_point[1]], [reference_point[2]], edgecolor="r", facecolor="red")


        reference_point = reference_points[1]
        print("reference point 2: ", reference_point)
        ax1.scatter3D([reference_point[0]], [reference_point[1]], [reference_point[2]], edgecolor="r", facecolor="red")


        ax1.view_init(30, -40)
        ax1.set_xlabel('$x_1$',fontsize=15)
        ax1.set_ylabel('$x_2$',fontsize=15)
        ax1.set_zlabel('$x_3$',fontsize=15)
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf", dpi=300)
        plt.show()
        print("DONE")
        
    else:
        # -- Implement the other option of using multiple gamma functions later --- #:
        print("TODO: Implement plotting of multiple gamma functions..")


if __name__ == '__main__':

    # --- Test to compare modulation implementations --- #
    gamma_type   = 0    # 0: single Gamma for all obstacles, 1: multiple independent Gammas
    dataset_type = '2D' # '2D': using the LASA 2D "joint" dataset, '3D': using the 3D Franka Environment
    which_data   = 0    # 0: lasa dataset, 1: 2D franka environment, 2: Random RBF generated environment 

    # Using environment/obstacles learned as svm-defined gamma functions
    if dataset_type == '2D':
        if (which_data == 0):
            # lower-bottom
            # x_target    = np.array([0.8, 0.2])
            # in-between obstacles
            x_target    = np.array([0.65, 0.65])

        if (which_data == 1):
            x,y,z     = rand_target_loc()
            x_target  = np.array([y, z])

        test2D_HBS_svmlearnedGammas(x_target, gamma_type, which_data)       

        # Using environment defined as sum of rbf functions gamma functions
        if (which_data == 2):
            test2D_HBS_rbf_env()

    if dataset_type == '3D':
        # x_target = np.array([0.8, 0.2])
        # x_target  = np.array([0.20, 0.65, 0.625])        
        x_target  = rand_target_loc()
        x_initial = np.array([0.516, -0.000, 1.221])
        test3D_HBS_svmlearnedGammas(x_target, x_initial, gamma_type)




# --- STATUS: TEST THIS LATEEEEERZZZ --- #
# def test2D_HBS_rbf_env():
#     '''demo of the HBS approach with RBF environment'''

#     # use the mouse
#     global mx, my
#     mx, my = None, None
#     def mouse_move(event):
#         global mx, my
#         mx, my = event.xdata, event.ydata

#     plt_resolution = 40

#     x_target = np.array([0.9, 0.8])
#     env = rbf_2d_env_svm.RBFEnvironment()
#     sim = rbf_2d_env_svm.PhysicsSimulator(env)

#     DRAW_INDIVIDUAL_GAMMAS = True
#     if DRAW_INDIVIDUAL_GAMMAS:
#         for gamma in env.individual_gammas:
#             plt.figure()
#             plt.axis([-1, 1, -1, 1])
#             # plt.imshow(gamma.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
#             X, Y       = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#             V, U       = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#             gamma_vals = np.ndarray(X.shape) 
#             for i in range(plt_resolution):
#                 for j in range(plt_resolution):
#                     x_query         = np.array([X[i, j], Y[i,j]])
#                     gamma_vals[i,j] = gamma(x_query)
#                     orig_ds         = modulation_svm.linear_controller(x_query, x_target)
#                     modulated_x_dot = modulation_svm.modulation_HBS(x_query, orig_ds, gammas=[gamma]) # * 0.15
#                     modulated_x_dot = modulated_x_dot / np.linalg.norm(modulated_x_dot +  epsilon) * 0.025
#                     U[i,j]     = modulated_x_dot[0]
#                     V[i,j]     = modulated_x_dot[1]    

#             plt.streamplot(X,Y,U,V, density = 6, linewidth=0.55, color='k')
#             levels    = np.array([0, 1, 2])
#             cs0       = plt.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#             cs        = plt.contourf(X, Y, gamma_vals, np.arange(-3, 3, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#             cbar      = plt.colorbar(cs)
#             cbar.add_lines(cs0)
#             plt.show()


#     x, y = -1, -1
#     sim.reset([x, y])
#     plt.figure()
#     plt.ion()
#     plt.connect('motion_notify_event', mouse_move)
#     plt.imshow(env.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
#     plt.axis([-1, 1, -1, 1])
#     plt.gca().set_aspect('equal')
#     plt.show()
#     agent_plot, = plt.plot([-1], [-1], 'C2o')

#     curr_loc = sim.agent_pos()

#     reading = env.lidar(sim.agent_pos())
#     plotted = rbf_2d_env.plot_lidar(curr_loc, reading)

#     X, Y = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#     V, U = np.meshgrid(np.linspace(-1, 1, plt_resolution), np.linspace(-1, 1, plt_resolution))
#     gamma_vals = np.ndarray(X.shape)
#     for i in range(plt_resolution):
#         for j in range(plt_resolution):
#             x_query = np.array([X[i, j], Y[i,j]])
#             gamma_vals[i,j] = min([gamma(x_query) for gamma in env.individual_gammas])
#             orig_ds         = modulation_svm.linear_controller(x_query, x_target)
#             modulated_x_dot = modulation_svm.modulation_HBS(x_query, orig_ds, gammas=env.individual_gammas) # * 0.15
#             modulated_x_dot = modulated_x_dot / np.linalg.norm(modulated_x_dot + epsilon) * 0.025
#             U[i,j]     = modulated_x_dot[0]
#             V[i,j]     = modulated_x_dot[1]

#     plt.streamplot(X,Y,U,V, density = 6, linewidth=0.55, color='k')
#     levels    = np.array([0, 1])
#     cs0       = plt.contour(X, Y, gamma_vals, levels, origin='lower', colors='k', linewidths=1)
#     cs        = plt.contourf(X, Y, gamma_vals, np.arange(-3, 3, 2), cmap=plt.cm.coolwarm, extend='both', alpha=0.8)
#     cbar      = plt.colorbar(cs)
#     cbar.add_lines(cs0)
#     plt.show()

#     while True:
#         if mx is not None and my is not None:
#             curr_loc = sim.agent_pos()
#             orig_ds  = modulation_svm.linear_controller(curr_loc, x_target)
#             d        = modulation_svm.modulation_HBS(curr_loc, orig_ds, gammas=env.individual_gammas)
#             d        = d / np.linalg.norm(d) * 0.03
#             sim.step(d)
#             min_gamma =  min([gamma(x_query) for gamma in env.individual_gammas])
#             print("min_gamma:", min_gamma)
#             print("ds:", d)
#             x, y = sim.agent_pos()
#             agent_plot.set_data([x], [y])
#             reading = env.lidar([x, y])
#             plotted = rbf_2d_env.plot_lidar([x, y], reading, plotted)
#         for indv_gamma in env.individual_gammas:
#             pt_x, pt_y = indv_gamma.center
#             plt.plot(pt_x, pt_y, 'go', markersize=10)
#             # plt.plot(indv_gamma.boundary_points.T[0,:],indv_gamma.boundary_points.T[1,:], 'yo', markersize=2)

#         plt.plot(0.9,0.8, 'y*', markersize=60)
#         plt.pause(0.1)        