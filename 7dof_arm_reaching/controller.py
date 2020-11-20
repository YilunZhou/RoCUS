import numpy as np
import torch
from reinforcement_learning.ppo import ActorCritic
from rapidly_exploring_random_tree.rrt import *

class RLController():
    def __init__(self, model_fn='reinforcement_learning/best.pt', device='cuda'):
        ckpt = torch.load(model_fn)
        activation = ckpt['config'].activation
        tanh_end = ckpt['config'].tanh_end
        self.actor_critic = ActorCritic(22, 7, activation=activation, tanh_end=tanh_end).to(device)
        self.actor_critic.load_state_dict(ckpt['actor-critic'])
        self.device = device

    def select_used(self, s):
        # j1-j7, end-effector xyz
        return s[:10]

    def get_trajectory(self, env, kernel=None):
        s = env.s()
        done = False
        traj = [self.select_used(s)]
        while not done:
            a = self.actor_critic.act(s, train=False)
            s, _, done, _ = env.step(a)
            traj.append(self.select_used(s))
        return np.array(traj)

    def to(self, device):
        self.device = device
        self.actor_critic.to(device)

    def log_prior(self, kernel):
        return 0


class RRTController():
    def __init__(self, control_res=0.05, collision_res=0.01):
        self.rrt_utils = RRTUtils()
        self.control_res = control_res
        self.collision_res = collision_res

    def attempt_connection(self, p, tree, neighbors, from_point):
        if from_point == 'nearest':
            nearest_dict = neighbors.nearest(p)
            idx = nearest_dict['idx']
            dist = nearest_dict['dist']
            pt = nearest_dict['point']
        else:
            assert isinstance(from_point, int)
            idx = from_point
            pt = tree[idx].pt
            dist = np.linalg.norm(np.array(p) - pt)
        num_checks = int(dist / self.collision_res) + 1
        configs = np.linspace(p, pt, num_checks)
        if all(self.rrt_utils.in_free(cfg) for cfg in configs):
            neighbors.insert(p)
            tree.insert(p, idx)
            return True
        else:
            return False

    def rrt(self, target_loc, kernel):
        start = [0, -0.3, 0.0, -2, 0, 2.0, m.pi / 4]
        end = self.rrt_utils.target_loc_to_joint_config(target_loc)
        tree = Tree(start)
        neighbors = NearestNeighbor()
        neighbors.insert(start)
        if self.attempt_connection(end, tree, neighbors, from_point=-1):
            return tree.get_path_from_root(len(tree) - 1)
        for i in range(1000):
            p = kernel[i]
            if self.attempt_connection(p, tree, neighbors, from_point='nearest') and self.attempt_connection(end, tree,
                                                                                                             neighbors,
                                                                                                             from_point=-1):
                return tree.get_path_from_root(len(tree) - 1)
        return None

    def increase_resolution(self, traj):
        traj = np.array(traj)
        new_traj = [traj[0]]
        for start, end in zip(traj[:-1], traj[1:]):
            max_movement = (start - end).max()
            if max_movement < self.control_res:
                new_traj.append(end)
            else:
                num_cuts = int(max_movement / self.control_res) + 1
                segments = np.linspace(start, end, num_cuts + 1)[1:]
                new_traj.extend(list(segments))
        new_traj = np.array(new_traj)
        return new_traj

    def get_trajectory(self, env, kernel):
        try:
            s = env.s()
            target_loc = s[16:19]
            planned_traj = self.rrt(target_loc, kernel)
            if planned_traj is None:
                return None
            planned_traj = self.increase_resolution(planned_traj)
            actual_traj = [s[:10]]
            done = False
            t = 1
            while not done:
                a = (planned_traj[t] - s[:7]) / 0.05
                s, _, done, _ = env.step(a)
                actual_traj.append(s[:10])
                if t < len(planned_traj) - 1:
                    t += 1
            actual_traj = np.array(actual_traj)
            return actual_traj
        except:
            return None

    def log_prior(self, kernel):
        return 0