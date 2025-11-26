#!/usr/bin/env python
# coding=utf-8
"""
演示缓冲区
用于on-policy算法（PPO、A2C、REINFORCE）的预填充经验
"""

import numpy as np
from .experience_generator import generate_expert_experiences


class DemonstrationBuffer:
    """演示缓冲区，存储专家经验供on-policy算法学习"""
    
    def __init__(self, env, n_demonstrations=1000):
        """
        初始化演示缓冲区
        
        Args:
            env: 环境实例
            n_demonstrations: 演示经验数量
        """
        self.env = env
        self.n_demonstrations = n_demonstrations
        self.demonstrations = []
        self._load_demonstrations()
    
    def _load_demonstrations(self):
        """加载演示经验"""
        print(f"开始生成 {self.n_demonstrations} 个演示经验...")
        experiences = generate_expert_experiences(self.env, self.n_demonstrations, n_simulations=500)
        
        # 转换为演示格式（只保留状态和动作）
        for state, action, reward, next_state, terminated, truncated in experiences:
            self.demonstrations.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': terminated or truncated
            })
        
        print(f"成功加载 {len(self.demonstrations)} 个演示经验")
    
    def sample_demonstrations(self, n_samples):
        """
        采样演示经验
        
        Args:
            n_samples: 采样数量
            
        Returns:
            demonstrations: 演示经验列表
        """
        if len(self.demonstrations) == 0:
            return []
        
        n_samples = min(n_samples, len(self.demonstrations))
        indices = np.random.choice(len(self.demonstrations), n_samples, replace=False)
        return [self.demonstrations[i] for i in indices]
    
    def get_all(self):
        """获取所有演示经验"""
        return self.demonstrations

