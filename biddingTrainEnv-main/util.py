import numpy as np
import random

class PrioritizedExperienceReplay:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            self.priorities[len(self.memory) - 1] = self.max_priority
        else:
            # Replace the oldest memory with the new one
            idx = np.random.randint(0, self.capacity)
            self.memory[idx] = experience
            self.priorities[idx] = self.max_priority

    def sample(self, batch_size):
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), size=4040, p=probs,replace=True)
        indices2 = np.random.choice(len(self.memory), size=1000, p=probs,replace=True)
        indices = np.append(indices,indices2)
        batch = [self.memory[i] for i in indices]
        weights = (len(self.memory) * probs[indices]) ** (-self.alpha)
        weights /= weights.max()
        return batch, weights
    def update_priorities(self, batch):
        for i, (_, _, reward, _, _) in enumerate(batch):
            td_error = abs(reward - self.max_priority)
            self.priorities[i] = td_error

# Example usage:
# Initialize the PrioritizedExperienceReplay with a capacity
# per = PrioritizedExperienceReplay(capacity=1000)
#
# Add experiences to the memory
# for i in range(1000):
#     experience = (np.array([1.]), np.array([1.]), np.array([1.]), np.array([1.]), np.array([1.]))
#     per.add(experience)
#
# # Sample a batch of experiences
# batch, weights = per.sample(batch_size=32)
#
# # Update priorities after learning step
# per.update_priorities(batch)


# 假设self.memory是一个包含经验元组的列表
# 例如：self.memory = [(state, action, reward, next_state, done), ...]

# 创建优先经验回放实例
# per = PrioritizedExperienceReplay(self.memory)
#
# # 采样batch_size个样本
# batch = per.sample(batch_size)