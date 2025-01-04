import numpy as np
import torch
from run.run_iql import run_iql

torch.manual_seed(2024)
np.random.seed(2024)

#1
#0209

if __name__ == "__main__":
    """程序主入口，运行IQL算法"""
    run_iql()
