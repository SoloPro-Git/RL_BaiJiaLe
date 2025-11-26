#!/usr/bin/env python
# coding=utf-8
"""
PPO (Proximal Policy Optimization) Agent 实现
On-policy 策略梯度方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent


class PPOBuffer:
    """PPO 经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def push(self, state, action, reward, log_prob, value, done):
        """存储样本"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def get_all(self):
        """获取所有数据"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.log_probs),
            np.array(self.values),
            np.array(self.dones)
        )


class PPOAgent(BaseAgent):
    """
    PPO Agent 实现
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', config=None):
        """
        初始化 PPO Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
            config: 配置字典，包含以下参数：
                - gamma: 折扣因子 (default: 0.99)
                - gae_lambda: GAE lambda参数 (default: 0.95)
                - lr: 学习率 (default: 0.0003)
                - clip_epsilon: PPO裁剪参数 (default: 0.2)
                - value_coef: 价值损失系数 (default: 0.5)
                - entropy_coef: 熵系数 (default: 0.01)
                - max_grad_norm: 梯度裁剪 (default: 0.5)
                - update_epochs: 每次更新的epoch数 (default: 4)
                - hidden_dims: 隐藏层维度列表 (default: [512, 512, 512])
        """
        super().__init__(state_dim, action_dim, device)
        
        # 默认配置
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.gae_lambda = config.get('gae_lambda', 0.95) if config else 0.95
        self.lr = config.get('lr', 0.0003) if config else 0.0003
        self.clip_epsilon = config.get('clip_epsilon', 0.2) if config else 0.2
        self.value_coef = config.get('value_coef', 0.5) if config else 0.5
        self.entropy_coef = config.get('entropy_coef', 0.01) if config else 0.01
        self.max_grad_norm = config.get('max_grad_norm', 0.5) if config else 0.5
        self.update_epochs = config.get('update_epochs', 4) if config else 4
        hidden_dims = config.get('hidden_dims', [512, 512, 512]) if config else [512, 512, 512]
        dropout = config.get('dropout', 0.1) if config else 0.1
        
        # 导入网络结构
        from networks.mlp import ActorCritic
        
        # 创建Actor-Critic网络
        self.ac_network = ActorCritic(state_dim, action_dim, hidden_dims, dropout).to(device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.buffer = PPOBuffer()
    
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
        
        if training:
            with torch.no_grad():
                action, log_prob, value = self.ac_network.get_action_and_value(state_tensor)
            return action.item(), log_prob.item(), value.item()
        else:
            with torch.no_grad():
                action_probs, value = self.ac_network(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
            return action
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """
        计算广义优势估计 (GAE)
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            dones: 终止标志序列
            next_value: 下一个状态的价值
            
        Returns:
            advantages: 优势值
            returns: 回报值
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self):
        """
        更新策略
        """
        if len(self.buffer.states) == 0:
            return
        
        # 获取所有数据
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get_all()
        
        # 计算最后一个状态的价值
        last_state = states[-1]
        last_state_tensor = torch.tensor(last_state, device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.ac_network(last_state_tensor)
            next_value = next_value.item()
        
        # 计算GAE和returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        
        # 多次更新
        for _ in range(self.update_epochs):
            # 前向传播
            action_probs, values_pred = self.ac_network(states_tensor)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # 计算策略损失（PPO裁剪）
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path + 'ppo_checkpoint.pth')
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path + 'ppo_checkpoint.pth', map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

