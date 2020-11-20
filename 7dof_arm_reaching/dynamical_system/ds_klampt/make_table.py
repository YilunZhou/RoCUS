import sys, pickle
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from sampler import ee_avg_jerk

warmup_discard = 2000
grid_size = 10
num_display = grid_size * grid_size

results = [
            ["ds_min_avg_jerk.samples.pkl"],
         ]
metrics = [
            ee_avg_jerk
          ]
priors  = [
            # "rrt_prior.pkl",
            # "ds_prior.pkl",
            "ds_prior.pkl"
          ]

metric_idx = 0
for result_set in results:
    for idx in range(len(result_set)):
        metric = metrics[metric_idx]

        prior_data = pickle.load(open(priors[idx], 'rb'))
        prior_envs = [t[0] for t in prior_data if t is not None]# and len(t) < 201]
        prior_trajs = [t[2] for t in prior_data if t is not None]# and len(t) < 201]

        prior_behavior_list = []
        for behavior_idx in range(500):
            traj = prior_trajs[behavior_idx]
            env = prior_envs[behavior_idx]
            behavior, acceptable = metric(traj, env)

            if acceptable:
                prior_behavior_list.append(behavior)

        print ("PRIOR", priors[idx], result_set[idx], np.mean(np.array(prior_behavior_list)))


        result = pickle.load(open(result_set[idx], 'rb'))
        env_samples = result['env']
        controller_samples = result['controller']
        trajectories = result['trajectory']
        behaviors = result['behavior']
        sample_idxs = list(range(len(env_samples)))[warmup_discard:]

        interval = int(len(sample_idxs) / num_display)
        sample_idxs = sample_idxs[::interval]
        sampled_behavior_list = []
        for behavior_idx in range(min(len(trajectories),500)):
            traj = trajectories[behavior_idx]
            env = env_samples[behavior_idx]
            sampled_behavior_list.append(metric(traj,env)[0])
            # env = RBFEnvironmentFromPoints(env_samples[idx])
            # env.draw()
            # plt.plot(traj[:, 0], traj[:, 1], 'y', alpha=0.6)
            # plt.axis([-1.2, 1.2, -1.2, 1.2])
            # plt.gca().set_aspect('equal')
            # plt.xticks([])
            # plt.yticks([])
            # # plt.savefig('min_center_deviation.pdf', bbox_inches='tight')
            # plt.show()
        print ("FINAL", result_set[idx], np.mean(sampled_behavior_list))


    metric_idx += 1
