# %%
import gym
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# %%


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(3, 64)
        self.affine2 = nn.Linear(64, 128)
        self.affine2
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(64, 1)
        self.affine2_ = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        mu = self.affine2(x)
        sigma = self.affine2_(x)
        sigma = F.softplus(sigma)
        # action_scores = self.affine2(x)
        # return F.softmax(action_scores, dim=1)
        return mu, sigma


# %%
env = gym.make('Pendulum-v0')
# policy = Policy()
# optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()  # 一个极小值
gamma = 0.9
render = False
log_interval = 100

# %%


class REINFORCE:
    def __init__(self) -> None:
        self.policy = Policy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-2)
        self.saved_log_probs = list()  # 存储ln \pi(\theta)
        self.rewards = list()  # 存储路径上的rewards

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma = self.policy(state)
        dists = torch.distributions.Normal(mu, sigma)
        action = dists.sample()
        probs = dists.log_prob(action)
        self.saved_log_probs.append(probs)
        return action.numpy()

    def learn(self):
        """ 
        self.rewards存储一条track上的reward
        """
        R = 0
        returns = list()  # 存储路径上t时刻之后的折扣总回报
        policy_loss = list()  # 存储路径上t时刻的loss
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.append(R)
        returns = list(reversed(returns))   # 阶段累计奖励G_t
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)  # baseline处理
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        # 清除这条路径的信息
        self.rewards.clear()
        self.saved_log_probs.clear()

# %%


def main():
    rl = REINFORCE()
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = rl.select_action(state)
            state, reward, done, _ = env.step(action)
            rl.rewards.append(reward)
            ep_reward += reward
            if render:
                env.render()
            if done:
                break
        rl.learn()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


# %%
main()
# %%
