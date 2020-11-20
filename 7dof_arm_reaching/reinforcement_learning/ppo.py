
import os, argparse
from copy import deepcopy as copy
from tqdm import trange

import numpy as np

import torch
from torch import nn
from torch.distributions import Normal, LogNormal
from torch.utils.data import DataLoader

import gym, pybulletgym

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, activation='relu', tanh_end=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        super(ActorCritic, self).__init__()
        if activation == 'relu':
            activ = nn.ReLU
        elif activation == 'tanh':
            activ = nn.Tanh
        modules = [nn.Linear(state_dim, 200),
		           activ(),
		           nn.Linear(200, 200),
		           activ(),
		           nn.Linear(200, 200),
		           activ(),
		    	   nn.Linear(200, action_dim)
          		  ]
        if tanh_end:
        	modules.append(nn.Tanh())
        self.action_mean = nn.Sequential(*modules)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim, ))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 200),
            activ(),
            nn.Linear(200, 200),
            activ(),
            nn.Linear(200, 200),
            activ(),
            nn.Linear(200, 1)
        )

    def act(self, state, train=True, memory=None):
        device = next(self.parameters()).device
        with torch.no_grad():
            assert len(state.shape)==1, 'only support a single state'
            state = torch.from_numpy(state).float().to(device)
            action_mean = self.action_mean(state)
            if not train:
                return action_mean.cpu().numpy()
            action_std = self.action_log_std.exp()
            action_dist = Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()
            memory.states.append(state.cpu().numpy())
            memory.actions.append(action.cpu().numpy())
            memory.logprobs.append(log_prob.item())
            return action.cpu().numpy()
    
    def evaluate(self, state, action):
        action_mean = self.action_mean(state)
        action_std = self.action_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)
        action_logprobs = dist.log_prob(action).sum(dim=1)
        dist_entropy = dist.entropy().mean() # sum(dim=1).mean()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, env, gamma, K_epochs, eps_clip, reward_avg, value_loss_coef, entropy_reg_coef, batchsize, activation, tanh_end, lr, device, 
    			 **kwargs):
        print(f'warning: additional kwargs not used: {list(kwargs.keys())}')
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.reward_avg = reward_avg
        self.policy = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], activation, tanh_end).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], activation, tanh_end).to(device)
        self.policy_old.load_state_dict(copy(self.policy.state_dict()))
        self.value_loss = nn.MSELoss()
        self.value_loss_coef = value_loss_coef
        self.entropy_reg_coef = entropy_reg_coef
        self.batchsize = batchsize
        self.device = device
        self.memory = Memory()

    def train_once(self, num_episode_per_train):
        assert len(self.memory.states) == 0, 'non-empty memory at the beginning of learning?'
        Rs = []
        for _ in range(num_episode_per_train):
            state = self.env.reset()
            R = 0
            done = False
            while not done:
                action = self.policy_old.act(state, train=True, memory=self.memory)
                state, reward, done, _ = self.env.step(action)
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)
                R += reward
            Rs.append(R)
        self.learn()
        return np.mean(Rs)
    
    def learn(self):
        # Monte Carlo estimate of state value:
        cum_rewards = []
        cur_cum_reward = 0
        for s, r, d in zip(reversed(self.memory.states), reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if d:
                cur_cum_reward = 0
            cur_cum_reward = r + (self.gamma * cur_cum_reward)
            cum_rewards.append(cur_cum_reward)
        cum_rewards = np.array(cum_rewards[::-1])
        
        # Normalizing the state value:
        if self.reward_avg:
            cum_rewards = (cum_rewards - cum_rewards.mean()) / (cum_rewards.std() + 1e-5)
        
        dataset = list(zip(cum_rewards, self.memory.states, self.memory.actions, self.memory.logprobs))
        dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for cum_rewards, old_states, old_actions, old_logprobs in dataloader:
                self.optimizer.zero_grad()
                # Evaluating old actions and values:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states.to(self.device), old_actions.to(self.device))
                # Finding the ratio (pi_theta / pi_theta_old):
                ratios = torch.exp(logprobs - old_logprobs.to(self.device))
                # Finding Surrogate Loss:
                advantages = cum_rewards.to(self.device) - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - torch.min(surr1, surr2)
                loss = (- torch.min(surr1, surr2)
                        + self.value_loss_coef * self.value_loss(state_values, cum_rewards.to(self.device)) 
                        - self.entropy_reg_coef * dist_entropy)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(copy(self.policy.state_dict()))
        self.memory.clear()

    def test(self, num_episode):
        Rs = []
        for _ in range(num_episode):
            state = self.env.reset()
            R = 0
            done = False
            while not done:
                action = self.policy.act(state, train=False)
                state, reward, done, _ = self.env.step(action)
                R += reward
            Rs.append(R)
        return np.mean(Rs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-total-episode', type=int, default=10000000)
    parser.add_argument('--num-episode-per-train', type=int, default=20)
    parser.add_argument('--num-episode-per-test', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--K-epochs', type=int, default=4)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--reward-avg', action='store_true')
    parser.set_defaults(reward_avg=True)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--entropy-reg-coef', type=float, default=0.01)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--tanh-end', action='store_true')
    parser.set_defaults(tanh_end=False)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save-dir', type=str)  # !!!!!
    parser.add_argument('--env-name', type=str)  # !!!!!
    config = parser.parse_args()
    print(config)
    assert config.save_dir is not None and config.env_name is not None

    env = gym.make(config.env_name)
    assert not os.path.isdir(config.save_dir)
    os.makedirs(config.save_dir)
    log_file = open(os.path.join(config.save_dir, 'progress.log'), 'w')
    ppo = PPO(env, **vars(config))
    best_test_R = float('-inf')
    for train_iter in trange(int(config.num_total_episode / config.num_episode_per_train)):
        train_R = ppo.train_once(config.num_episode_per_train)
        log_file.write(f'Train reward: {train_R}\n')
        if (train_iter + 1) % 10 == 0:
            test_R = ppo.test(config.num_episode_per_test)
            log_file.write(f'Test reward: {test_R}\n')
            if test_R > best_test_R:
                torch.save({'actor-critic': ppo.policy.state_dict(), 
                            'optimizer': ppo.optimizer.state_dict(), 
                            'test_R': test_R, 'config': config}, os.path.join(config.save_dir, 'best.pt'))
                best_test_R = test_R
                log_file.write('Saved best model so far\n')
            if (train_iter + 1) % 1000 == 0:
                torch.save({'actor-critic': ppo.policy.state_dict(), 
                            'optimizer': ppo.optimizer.state_dict(), 
                            'test_R': test_R, 'config': config}, os.path.join(config.save_dir, f'iter_{train_iter + 1}.pt'))
        log_file.flush()
if __name__ == '__main__':
    main()
