import numpy as np
from scipy.stats import norm, truncnorm

class TransitionKernel():
    '''
	A transition kernel on a random variable (or a set of RVs) stores the current value of the RV,
	propose() will propose a new RV by setting the value attribute, and return forward and backward transition log probability.
	revert() will revert the proposed value. revert can only be done once after a proposal.
	sample_prior will reset the current value to one sampled from the prior, and erase prev_value to None since the chain is broken.
	'''

    def __init__(self):
        self.sample_prior()

    def propose(self):
        self.prev_value = self.value
        self.value = 0
        return 0, 0

    def revert(self):
        assert self.prev_value is not None, 'no previous value available'
        self.value = self.prev_value
        self.prev_value = None

    def sample_prior(self):
        self.value = 0
        self.prev_value = None


class ReachingEnvKernelUniform(TransitionKernel):
    def __init__(self, left_lower=[-0.5, -0.3, 0.65], left_upper=[-0.05, 0.2, 1.0],
                 right_lower=[0.05, -0.3, 0.65], right_upper=[0.5, 0.2, 1.0]):
        self.left_lower = left_lower
        self.left_upper = left_upper
        self.right_lower = right_lower
        self.right_upper = right_upper
        super(ReachingEnvKernelUniform, self).__init__()

    def propose(self):
        self.prev_value = self.value
        use_left = np.random.random() < 0.5
        if use_left:
            self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
        else:
            self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
        return 0, 0

    def sample_prior(self):
        use_left = np.random.random() < 0.5
        if use_left:
            self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
        else:
            self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
        self.prev_value = None


def truncnorm_rvs(a, b, mean, std):
    a_use = (a - mean) / std
    b_use = (b - mean) / std
    return truncnorm.rvs(a_use, b_use, mean, std)


def truncnorm_logpdf(x, a, b, mean, std):
    a_use = (a - mean) / std
    b_use = (b - mean) / std
    return truncnorm.logpdf(x, a_use, b_use, mean, std)


class ReachingEnvKernelNormal(TransitionKernel):
    def __init__(self, left_lower=[-0.5, -0.3, 0.65], left_upper=[-0.05, 0.2, 1.0],
                 right_lower=[0.05, -0.3, 0.65], right_upper=[0.5, 0.2, 1.0],
                 sigma_x=0.1, sigma_y=0.05, sigma_z=0.035):
        assert left_lower[1] == right_lower[1] and left_lower[2] == right_lower[2]
        assert left_upper[1] == right_upper[1] and left_upper[2] == right_upper[2]
        self.left_lower = left_lower
        self.left_upper = left_upper
        self.right_lower = right_lower
        self.right_upper = right_upper
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        super(ReachingEnvKernelNormal, self).__init__()

    def propose(self):
        self.prev_value = self.value
        cur_x, cur_y, cur_z = self.value
        total_forward_log_prob = 0
        total_backward_log_prob = 0

        left_total_prob = norm.cdf(self.left_upper[0], loc=cur_x, scale=self.sigma_x) - norm.cdf(self.left_lower[0],
                                                                                                 loc=cur_x,
                                                                                                 scale=self.sigma_x)
        right_total_prob = norm.cdf(self.right_upper[0], loc=cur_x, scale=self.sigma_x) - norm.cdf(self.right_lower[0],
                                                                                                   loc=cur_x,
                                                                                                   scale=self.sigma_x)
        left_ratio = left_total_prob / (left_total_prob + right_total_prob)
        if np.random.random() < left_ratio:
            prop_x = truncnorm_rvs(self.left_lower[0], self.left_upper[0], cur_x, self.sigma_x)
            total_forward_log_prob += np.log(left_ratio) + truncnorm_logpdf(prop_x, self.left_lower[0],
                                                                            self.left_upper[0], cur_x, self.sigma_x)
        else:
            prop_x = truncnorm_rvs(self.right_lower[0], self.right_upper[0], cur_x, self.sigma_x)
            total_forward_log_prob += np.log(1 - left_ratio) + truncnorm_logpdf(prop_x, self.right_lower[0],
                                                                                self.right_upper[0], cur_x,
                                                                                self.sigma_x)

        back_left_total_prob = norm.cdf(self.left_upper[0], loc=prop_x, scale=self.sigma_x) - norm.cdf(
            self.left_lower[0], loc=prop_x, scale=self.sigma_x)
        back_right_total_prob = norm.cdf(self.right_upper[0], loc=prop_x, scale=self.sigma_x) - norm.cdf(
            self.right_lower[0], loc=prop_x, scale=self.sigma_x)
        back_left_ratio = back_left_total_prob / (back_left_total_prob + back_right_total_prob)

        assert self.left_lower[0] <= cur_x <= self.left_upper[0] or self.right_lower[0] <= cur_x <= self.right_upper[0]
        if self.left_lower[0] <= cur_x <= self.left_upper[0]:
            total_backward_log_prob += np.log(back_left_ratio) + truncnorm_logpdf(cur_x, self.left_lower[0],
                                                                                  self.left_upper[0], prop_x,
                                                                                  self.sigma_x)
        else:
            total_backward_log_prob += np.log(1 - back_left_ratio) + truncnorm_logpdf(cur_x, self.right_lower[0],
                                                                                      self.right_upper[0], prop_x,
                                                                                      self.sigma_x)

        prop_y = truncnorm_rvs(self.left_lower[1], self.left_upper[1], cur_y, self.sigma_y)
        total_forward_log_prob += truncnorm_logpdf(prop_y, self.left_lower[1], self.left_upper[1], cur_y, self.sigma_y)
        total_backward_log_prob += truncnorm_logpdf(cur_y, self.left_lower[1], self.left_upper[1], prop_y, self.sigma_y)

        prop_z = truncnorm_rvs(self.left_lower[2], self.left_upper[2], cur_z, self.sigma_z)
        total_forward_log_prob += truncnorm_logpdf(prop_z, self.left_lower[2], self.left_upper[2], cur_z, self.sigma_z)
        total_backward_log_prob += truncnorm_logpdf(cur_z, self.left_lower[2], self.left_upper[2], prop_z, self.sigma_z)

        self.value = [prop_x, prop_y, prop_z]
        return total_forward_log_prob, total_backward_log_prob

    def sample_prior(self):
        use_left = np.random.random() < 0.5
        if use_left:
            self.value = np.random.uniform(low=self.left_lower, high=self.left_upper)
        else:
            self.value = np.random.uniform(low=self.right_lower, high=self.right_upper)
        self.prev_value = None


class RRTKernelNormal(TransitionKernel):
    def __init__(self, cspace_low=[-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671],
                 cspace_high=[2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671], sigma_ratio=0.1):
        self.cspace_low = np.array(cspace_low)
        self.cspace_high = np.array(cspace_high)
        self.sigma = (np.array(cspace_high) - cspace_low) * sigma_ratio
        super(RRTKernelNormal, self).__init__()

    def propose(self):
        self.prev_value = self.value
        total_forward_log_prob = 0
        total_backward_log_prob = 0
        self.value = []
        for pv in self.prev_value:
            v = np.zeros(pv.shape)
            for i, p_val in enumerate(pv):
                v[i] = truncnorm_rvs(a=self.cspace_low[i], b=self.cspace_high[i], mean=p_val, std=self.sigma[i])
                total_forward_log_prob += truncnorm_logpdf(v[i], a=self.cspace_low[i], b=self.cspace_high[i],
                                                           mean=p_val, std=self.sigma[i])
                total_backward_log_prob += truncnorm_logpdf(p_val, a=self.cspace_low[i], b=self.cspace_high[i],
                                                            mean=v[i], std=self.sigma[i])
            self.value.append(v)
        return total_forward_log_prob, total_backward_log_prob

    def sample_prior(self):
        self.value = []
        self.prev_value = None

    def __getitem__(self, idx):
        if idx >= len(self.value):
            if idx > len(self.value):
                print('accessing non-consecutive entries? ')
            for _ in range(len(self.value), idx + 1):
                new = np.random.uniform(low=self.cspace_low, high=self.cspace_high)
                self.value.append(new)
        return self.value[idx]

    def __setitem__(self, idx, val):
        raise Exception('You should not mannually set kernel entries. ')
