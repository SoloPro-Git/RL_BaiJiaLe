#!/usr/bin/env python
# coding=utf-8
"""
DQN (Deep Q-Network) Agent 实现
Off-policy 值函数方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from agents.base_agent import BaseAgent


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, terminated, truncated):
        """
        存储转移样本
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            terminated: 是否自然终止
            truncated: 是否被截断
        """
        done = terminated or truncated
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """随机采样一批样本"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent 实现
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', config=None):
        """
        初始化 DQN Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
            config: 配置字典，包含以下参数：
                - gamma: 折扣因子 (default: 0.99)
                - epsilon_start: 初始探索率 (default: 0.99)
                - epsilon_end: 最终探索率 (default: 0.005)
                - epsilon_decay: 探索率衰减率 (default: 500)
                - lr: 学习率 (default: 0.0001)
                - memory_capacity: 经验回放容量 (default: 10000)
                - batch_size: 批次大小 (default: 128)
                - target_update: 目标网络更新频率 (default: 4)
                - hidden_dims: 隐藏层维度列表 (default: [512, 512, 512])
        """
        super().__init__(state_dim, action_dim, device)
        
        # 默认配置
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.epsilon_start = config.get('epsilon_start', 0.99) if config else 0.99
        self.epsilon_end = config.get('epsilon_end', 0.005) if config else 0.005
        self.epsilon_decay = config.get('epsilon_decay', 500) if config else 500
        self.lr = config.get('lr', 0.0001) if config else 0.0001
        self.memory_capacity = config.get('memory_capacity', 10000) if config else 10000
        self.batch_size = config.get('batch_size', 128) if config else 128
        self.target_update = config.get('target_update', 4) if config else 4
        hidden_dims = config.get('hidden_dims', [512, 512, 512]) if config else [512, 512, 512]
        
        # 导入网络结构
        from networks.mlp import MLP
        
        # 创建网络
        self.policy_net = MLP(state_dim, action_dim, hidden_dims).to(device)
        self.target_net = MLP(state_dim, action_dim, hidden_dims).to(device)
        
        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 经验回放
        self.memory = ReplayBuffer(self.memory_capacity)
        
        # 训练计数器
        self.frame_idx = 0
        self.update_count = 0
    
    def select_action(self, state, training=True):
        """
        选择动作（epsilon-greedy策略）
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
            
        Returns:
            action: 选择的动作
        """
        if training:
            self.frame_idx += 1
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                     math.exp(-1. * self.frame_idx / self.epsilon_decay)
            
            if random.random() > epsilon:
                # 利用：选择Q值最大的动作
                with torch.no_grad():
                    state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            else:
                # 探索：随机选择动作
                action = random.randrange(self.action_dim)
        else:
            # 评估模式：直接选择最优动作
            with torch.no_grad():
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        return action
    
    def update(self, times=1):
        """
        更新网络参数
        
        Args:
            times: 更新次数
        """
        if len(self.memory) < self.batch_size:
            return
        
        for _ in range(times):
            # 采样批次
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
                self.memory.sample(self.batch_size)
            
            # 转换为tensor
            state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float32)
            action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
            reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float32)
            next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float32)
            done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float32)
            
            # 计算当前Q值
            q_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # 计算目标Q值
            with torch.no_grad():
                next_q_values = self.target_net(next_state_batch).max(1)[0]
                expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
            # 计算损失
            loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
            
            # 更新网络
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
            self.update_count += 1
            
            # 更新目标网络
            if self.update_count % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_idx': self.frame_idx,
        }, path + 'dqn_checkpoint.pth')
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path + 'dqn_checkpoint.pth', map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.frame_idx = checkpoint.get('frame_idx', 0)

