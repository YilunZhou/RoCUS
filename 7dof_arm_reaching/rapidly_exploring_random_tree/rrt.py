from ik import IK
import gym, pybulletgym
import math as m
import pickle, os, time
from copy import deepcopy as copy
from multiprocessing import Pool
from tqdm import tqdm, trange

import numpy as np
from scipy.stats import norm, truncnorm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys


class RRTUtils():
    def __init__(self):
        self.ik = IK()
        self.plan_env = gym.make('PandaReacher-v0', shelf=True)
        self.plan_env.reset()
        self.p = self.plan_env.robot._p
        self.drive_res = 100

    def getConfig(self):
        config = self.p.getJointStates(0, range(7))
        config = np.array([e[0] for e in config])
        return config

    def setConfig(self, cfg):
        for j in range(7):
            self.p.resetJointState(0, j, cfg[j], 0)
        self.p.setJointMotorControlArray(bodyIndex=0, jointIndices=range(7), controlMode=pybullet.POSITION_CONTROL,
                                         targetPositions=cfg, positionGains=[0.5] * 7, velocityGains=[1.0] * 7)

    def drive_to_position(self, from_joint_config, to_joint_config):
        waypoints = np.linspace(from_joint_config, to_joint_config, self.drive_res)
        self.setConfig(from_joint_config)
        for waypoint in waypoints:
            self.p.setJointMotorControlArray(bodyIndex=0, jointIndices=range(7), controlMode=pybullet.POSITION_CONTROL,
                                             targetPositions=waypoint, positionGains=[0.5] * 7, velocityGains=[1.0] * 7)
            self.p.stepSimulation()
        end_config = self.getConfig()
        if not self.in_collision(end_config):
            return end_config
        for u in np.arange(0.01, 1, 0.01):
            target_config = (1 - u) * end_config + u * np.array(from_joint_config)
            if not self.in_collision(target_config):
                return target_config
        raise Exception('not returning', target_loc)

    def in_collision(self, config):
        self.setConfig(config)
        self.p.stepSimulation()
        contacts = self.p.getContactPoints()
        for c in contacts:
            _, bodyA, bodyB, linkA, linkB, posA, posB, norm, dist, nf, _, _, _, _ = c
            if dist < 0:
                return True
        return False

    def in_free(self, config):
        return not self.in_collision(config)

    def target_loc_to_joint_config(self, target_loc):
        flag, cfg, initial_cfg = self.ik.solve(target_loc)
        self.plan_env.reset(target_loc=target_loc)
        cfg = self.drive_to_position(initial_cfg, cfg)
        assert not self.in_collision(cfg), target_loc
        return cfg


class NearestNeighbor():
    def __init__(self):
        self.points = []

    def insert(self, pt):
        self.points.append(pt)

    def nearest(self, pt):
        dists = np.linalg.norm(np.array(self.points) - pt, axis=1)
        idx = np.argmin(dists)
        return {'idx': idx, 'dist': np.min(dists), 'point': self.points[idx]}


class Node():
    def __init__(self, pt, parent=None):
        self.pt = pt
        self.parent = parent


class Tree():
    def __init__(self, root_pt):
        self.root = Node(root_pt)
        self.nodes = [self.root]

    def insert(self, pt, parent_idx):
        self.nodes.append(Node(pt, self.nodes[parent_idx]))

    def get_path_from_root(self, idx):
        path = [self.nodes[idx]]
        while path[-1].parent is not None:
            path.append(path[-1].parent)
        path = np.array([p.pt for p in path[::-1]])
        return path

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]


