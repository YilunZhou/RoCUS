
import numpy as np
from scipy.interpolate import RectBivariateSpline
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

class RBFArena():
    '''
    a 2D arena defined as the sum of several positive RBF functions
    support Gamma function query (returns a value greater than 1 in free space, 
    1 on obstacle boundary, and smaller than 1 in obstacle)
    and its gradient (pointing "outward" toward free space)
    also supports lidar sensor simulation.
    '''
    def __init__(self, N_points=15, obs_low=-0.7, obs_high=0.7, rbf_gamma=25, threshold=0.9, 
                 grid_res=150, grid_bd=1.2):
        self.N_points  = N_points
        self.obs_low   = obs_low
        self.obs_high  = obs_high
        self.rbf_gamma = rbf_gamma
        self.threshold = threshold
        self.grid_res   = grid_res
        self.grid_bd = grid_bd
        self.reset()

    def reset(self, obs_override=None):
        if obs_override is None:
            self.points = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.N_points, 2))
        else:
            self.points = np.array(obs_override)
        self.occ_grid = None
        self.interpolator = None

    def gamma_batch(self, p):
        kernel_val = - rbf_kernel(self.points, p, self.rbf_gamma).sum(axis=0)
        return kernel_val + self.threshold + 1

    def gamma(self, p):
        return self.gamma_batch(np.array([p]))[0]

    def gamma_grad_batch(self, p):
        kernel_vals = rbf_kernel(self.points, p, gamma=self.rbf_gamma)
        coefs = 2 * self.rbf_gamma * np.array([[p_j - p_i for p_j in p] for p_i in self.points]).reshape(
            self.N_points, p.shape[0], 2)
        coefs = np.transpose(coefs, [2, 0, 1])
        grad_vals = kernel_vals * coefs  # 2 x N_points x p.shape[0]
        grad_vals = grad_vals.sum(axis=1).T  # p.shape[0] x 2
        return grad_vals

    def gamma_grad(self, p):
        return self.gamma_grad_batch(np.array([p]))[0]

    @property
    def occ_grid(self):
        if self.__occ_grid is None:
            xs = np.linspace(-self.grid_bd, self.grid_bd, self.grid_res)
            ys = np.linspace(-self.grid_bd, self.grid_bd, self.grid_res)
            xy_grid = np.stack([a.flatten() for a in np.meshgrid(xs, ys)], axis=1)
            self.__occ_grid = (self.gamma_batch(xy_grid) <= 1).astype('int').reshape(self.grid_res, self.grid_res)
        return self.__occ_grid

    @occ_grid.setter
    def occ_grid(self, occ_grid):
        self.__occ_grid = occ_grid

    @property
    def interpolator(self):
        if self.__interpolator is None:
            xs = np.linspace(-self.grid_bd, self.grid_bd, self.grid_res)
            ys = np.linspace(-self.grid_bd, self.grid_bd, self.grid_res)
            self.__interpolator = RectBivariateSpline(x=xs, y=ys, z=self.occ_grid)
        return self.__interpolator

    @interpolator.setter
    def interpolator(self, interpolator):
        self.__interpolator = interpolator

    def lidar_sensor(self, is_free_func, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
        dists = np.arange(dist_incr, max_range, dist_incr)
        thetas = np.linspace(0, 2 * np.pi, num_readings + 1)[:-1]
        lidar_xs = np.array([dists * np.cos(th) for th in thetas]).flatten()
        lidar_ys = np.array([dists * np.sin(th) for th in thetas]).flatten()
        x, y = p
        xs = lidar_xs + x
        ys = lidar_ys + y
        is_free = is_free_func(xs, ys).reshape(num_readings, -1).astype('int')
        idxs = np.argwhere(is_free==0)
        reading = {th_idx: max_range for th_idx in range(num_readings)}
        for th_idx, incr in idxs:
            if reading[th_idx] > (incr + 1) * dist_incr:
                reading[th_idx] = (incr + 1) * dist_incr
        return np.array([reading[th_idx] for th_idx in range(num_readings)])

    def lidar_fast(self, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
        is_free_func = lambda xs, ys: self.interpolator.ev(ys, xs) < 0.9
        return self.lidar_sensor(is_free_func, p, num_readings, max_range, dist_incr, theta)

    def lidar_exact(self, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
        is_free_func = lambda xs, ys: self.gamma_batch(np.stack([xs, ys], axis=1)) > 1
        return self.lidar_sensor(is_free_func, p, num_readings, max_range, dist_incr, theta)

    def lidar(self, p, mode='fast', num_readings=16, max_range=1, dist_incr=0.002, theta=0):
        if mode == 'fast':
            return self.lidar_fast(p, num_readings, max_range, dist_incr, theta)
        elif mode == 'exact':
            return self.lidar_exact(p, num_readings, max_range, dist_incr, theta)
        else:
            raise Exception(f'Unrecognized mode: {mode}')

class PhysicsSimulator():
    def __init__(self, arena):
        self.arena = arena

    def reset(self, p):
        assert self.is_free(*p), 'the reset position must be in free space'
        self.x = p[0]
        self.y = p[1]

    def agent_pos(self):
        return np.array([self.x, self.y])

    def is_collision(self, x, y):
        return self.arena.gamma([x, y]) <= 1

    def is_free(self, x, y):
        return not self.is_collision(x, y)

    def find_contact(self, x, y, dx, dy, iteration=5):
        '''find the contact point with obstacle of moving (x, y) in (dx, dy) direction by interval halving'''
        last_good = None
        last_good_u = None
        factor = 0.5
        u = 0.5
        for _ in range(iteration):
            cur_point = [x + dx * u, y + dy * u]
            factor = factor / 2
            if self.is_free(*cur_point):
                last_good = cur_point
                assert last_good_u is None or last_good_u < u
                last_good_u = u
                u = u + factor
            else:
                u = u - factor
        if last_good is None:
            last_good = [x, y]
            last_good_u = 0
        return last_good, last_good_u

    def step(self, dp, return_collide=False):
        assert self.is_free(self.x, self.y)
        dx, dy = dp
        if self.is_free(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy
            if return_collide:
                return np.array([self.x, self.y]), 0
            else:
                return np.array([self.x, self.y])
        # in collision
        # 1. find closest point without collision by interval halving
        contact, u = self.find_contact(self.x, self.y, dx, dy)
        # 2. projecting the remaining (dx, dy) along the tangent
        normal = self.arena.gamma_grad(contact)
        tangent = np.array([normal[1], -normal[0]])
        remaining = ([dx * (1 - u), dy * (1 - u)])
        proj = np.dot(tangent, remaining) / np.linalg.norm(tangent)**2 * tangent
        end = proj + contact
        # 3. if collision, then find again the contact point along the tangent direction
        if self.is_collision(*end):
            end, _ = self.find_contact(contact[0], contact[1], proj[0], proj[1])
        self.x = end[0]
        self.y = end[1]
        if return_collide:
            return np.array([self.x, self.y]), 1
        else:
            return np.array([self.x, self.y])


class RBF2dGymEnv():
    def __init__(self, time_limit=200, dxy_limit=0.03, oob_termination=True, use_lidar=True, **kwargs):
        self.arena = RBFArena(**kwargs)
        self.time_limit = time_limit
        self.dxy_limit = dxy_limit
        self.target_pos = [1, 1]
        self.oob_termination = oob_termination
        self.enable_lidar = use_lidar

    def reset(self, obs_override=None):
        self.arena.reset(obs_override)
        self.sim = PhysicsSimulator(self.arena)
        self.sim.reset([-1, -1])
        self.t = 0
        return self.s()

    def step(self, a):
        assert self.t < self.time_limit, 'time limit already exceeded'
        self.t += 1
        if self.dxy_limit is not None:
            a = np.clip(a, -1, 1) * self.dxy_limit
        self.sim.step(a)
        dist = np.linalg.norm(self.target_pos - self.sim.agent_pos())
        ax, ay = self.sim.agent_pos()
        r = - dist
        out_of_bound = not (-1.2 < ax < 1.2 and -1.2 < ay < 1.2)
        if out_of_bound:
            r = r - 500
        done = (self.t == self.time_limit or dist < 0.03)
        if self.oob_termination and out_of_bound:
            done = True
        return self.s(), r, done, None

    def agent_pos(self):
        return self.sim.agent_pos()

    def s(self):
        s = self.sim.agent_pos()
        if self.enable_lidar:
            s = np.concatenate([s, self.arena.lidar(s)])
        return s

    def log_prior(self, value):
        assert np.all(abs(value) <= 0.7)
        return 0
