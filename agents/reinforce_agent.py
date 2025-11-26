#!/usr/bin/env python
# coding=utf-8
"""
REINFORCE Agent 实现
经典策略梯度方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent


class REINFORCEBuffer:
    """REINFORCE 经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
    
    def push(self, state, action, reward, log_prob, done):
        """存储样本"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
    
    def get_all(self):
        """获取所有数据"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.log_probs),
            np.array(self.dones)
        )


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE Agent 实现
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', config=None):
        """
        初始化 REINFORCE Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
            config: 配置字典，包含以下参数：
                - gamma: 折扣因子 (default: 0.99)
                - lr: 学习率 (default: 0.0003)
                - entropy_coef: 熵系数 (default: 0.01)
                - max_grad_norm: 梯度裁剪 (default: 0.5)
                - hidden_dims: 隐藏层维度列表 (default: [512, 512, 512])
        """
        super().__init__(state_dim, action_dim, device)
        
        # 默认配置
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.lr = config.get('lr', 0.0003) if config else 0.0003
        self.entropy_coef = config.get('entropy_coef', 0.01) if config else 0.01
        self.max_grad_norm = config.get('max_grad_norm', 0.5) if config else 0.5
        hidden_dims = config.get('hidden_dims', [512, 512, 512]) if config else [512, 512, 512]
        dropout = config.get('dropout', 0.1) if config else 0.1
        
        # 导入网络结构
        from networks.mlp import PolicyNetwork
        
        # 创建策略网络
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims, dropout).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.buffer = REINFORCEBuffer()
    
    def select_action(self, state, training=True):
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
            
        Returns:
            action: 选择的动作
        """
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # 单样本推理时使用eval模式，避免BatchNorm报错
        self.policy_net.eval()
        if training:
            with torch.no_grad():
                action, log_prob = self.policy_net.get_action_and_log_prob(state_tensor)
            result = action.item(), log_prob.item()
        else:
            with torch.no_grad():
                action_probs = self.policy_net(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
            result = action
        self.policy_net.train()  # 恢复训练模式
        return result
    
    def compute_returns(self, rewards, dones):
        """
        计算折扣回报
        
        Args:
            rewards: 奖励序列
            dones: 终止标志序列
            
        Returns:
            returns: 回报值
        """
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def update(self):
        """
        更新策略
        """
        if len(self.buffer.states) == 0:
            return
        
        # 确保在训练模式下更新（BatchNorm需要batch统计）
        self.policy_net.train()
        
        # 获取所有数据
        states, actions, rewards, log_probs, dones = self.buffer.get_all()
        
        # 计算回报
        returns = self.compute_returns(rewards, dones)
        
        # 归一化回报（减少方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 转换为tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        old_log_probs_tensor = torch.tensor(log_probs, device=self.device, dtype=torch.float32)
        
        # 前向传播
        action_probs = self.policy_net(states_tensor)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        # 计算损失
        policy_loss = -(new_log_probs * returns_tensor).mean()
        loss = policy_loss - self.entropy_coef * entropy
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path + 'reinforce_checkpoint.pth')
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path + 'reinforce_checkpoint.pth', map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

