import random
from collections import namedtuple, deque
import numpy as np
import torch
from util import PrioritizedExperienceReplay

# 定义一个命名元组来存储经验
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
per = PrioritizedExperienceReplay(capacity=5040, alpha=0.6)

class ReplayBuffer:
    """
    强化学习储存训练数据的训练池
    """

    def __init__(self):
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        """保存一个经验元组"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """随机抽取一批经验"""
        for data in self.memory:
             per.add(data)
        batch, weights = per.sample(batch_size=batch_size)
        per.update_priorities(batch)
        tem,_ = per.sample(batch_size)

        # tem = random.sample(self.memory, batch_size)  # reward ！0：0 = 76% : 24%
        states, actions, rewards, next_states, dones = zip(*tem)

        states, actions, rewards, next_states, dones = np.stack(states), np.stack(actions), np.stack(rewards), np.stack(
            next_states), np.stack(dones)
        states, actions, rewards, next_states, dones = torch.FloatTensor(states), torch.FloatTensor(
            actions), torch.FloatTensor(rewards), torch.FloatTensor(next_states), torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回当前缓冲区的大小"""
        return len(self.memory)


# if __name__ == '__main__':
#     buffer = ReplayBuffer()
#     for i in range(1000):
#         buffer.push(np.array([1, 2, 3]), np.array(4), np.array(5), np.array([6, 7, 8]), np.array(0))
    # print(buffer.sample(20))
