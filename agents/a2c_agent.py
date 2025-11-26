#!/usr/bin/env python
# coding=utf-8
"""
A2C (Advantage Actor-Critic) Agent 实现
On-policy Actor-Critic方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.base_agent import BaseAgent


class A2CBuffer:
    """A2C 经验缓冲区"""
    
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


class A2CAgent(BaseAgent):
    """
    A2C Agent 实现
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', config=None, env=None):
        """
        初始化 A2C Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备
            config: 配置字典，包含以下参数：
                - gamma: 折扣因子 (default: 0.99)
                - lr: 学习率 (default: 0.0003)
                - value_coef: 价值损失系数 (default: 0.5)
                - entropy_coef: 熵系数 (default: 0.01)
                - max_grad_norm: 梯度裁剪 (default: 0.5)
                - n_steps: n步回报 (default: 5)
                - hidden_dims: 隐藏层维度列表 (default: [512, 512, 512])
                - prefill_experiences: 是否预填充演示经验 (default: False)
                - prefill_size: 预填充经验数量 (default: 1000)
            env: 环境实例（用于生成预填充经验）
        """
        super().__init__(state_dim, action_dim, device)
        
        # 默认配置
        self.gamma = config.get('gamma', 0.99) if config else 0.99
        self.lr = config.get('lr', 0.0003) if config else 0.0003
        self.value_coef = config.get('value_coef', 0.5) if config else 0.5
        self.entropy_coef = config.get('entropy_coef', 0.01) if config else 0.01
        self.max_grad_norm = config.get('max_grad_norm', 0.5) if config else 0.5
        self.n_steps = config.get('n_steps', 5) if config else 5
        hidden_dims = config.get('hidden_dims', [512, 512, 512]) if config else [512, 512, 512]
        dropout = config.get('dropout', 0.1) if config else 0.1
        
        # 导入网络结构
        from networks.mlp import ActorCritic
        
        # 创建Actor-Critic网络
        self.ac_network = ActorCritic(state_dim, action_dim, hidden_dims, dropout).to(device)
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.buffer = A2CBuffer()
        
        # 预填充演示经验（用于初始策略学习）
        prefill_experiences = config.get('prefill_experiences', False) if config else False
        prefill_size = config.get('prefill_size', 1000) if config else 1000
        
        if prefill_experiences and env is not None:
            print(f"开始为A2C加载 {prefill_size} 个演示经验...")
            from utils.demonstration_buffer import DemonstrationBuffer
            self.demonstration_buffer = DemonstrationBuffer(env, prefill_size)
            # 使用演示经验进行初始策略学习
            self._learn_from_demonstrations()
            print("演示经验学习完成")
        else:
            self.demonstration_buffer = None
    
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
        self.ac_network.eval()
        if training:
            with torch.no_grad():
                action, log_prob, value = self.ac_network.get_action_and_value(state_tensor)
            result = action.item(), log_prob.item(), value.item()
        else:
            with torch.no_grad():
                action_probs, value = self.ac_network(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
            result = action
        self.ac_network.train()  # 恢复训练模式
        return result
    
    def _learn_from_demonstrations(self, n_iterations=10):
        """
        从演示经验中学习初始策略
        
        Args:
            n_iterations: 学习迭代次数
        """
        if self.demonstration_buffer is None or len(self.demonstration_buffer.demonstrations) == 0:
            return
        
        demonstrations = self.demonstration_buffer.sample_demonstrations(100)
        
        for _ in range(n_iterations):
            if len(demonstrations) == 0:
                break
            
            states = np.array([d['state'] for d in demonstrations])
            actions = np.array([d['action'] for d in demonstrations])
            rewards = np.array([d['reward'] for d in demonstrations])
            
            states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            
            action_probs, values = self.ac_network(states_tensor)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            policy_loss = -log_probs.mean()
            value_loss = nn.MSELoss()(values.squeeze(), rewards_tensor)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def compute_returns(self, rewards, values, dones, next_value=0):
        """
        计算n步回报
        
        Args:
            rewards: 奖励序列
            values: 价值序列
            dones: 终止标志序列
            next_value: 下一个状态的价值
            
        Returns:
            returns: 回报值
            advantages: 优势值
        """
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # 计算n步回报
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            # n步回报
            returns[t] = rewards[t] + self.gamma * next_non_terminal * next_value_t
            advantages[t] = returns[t] - values[t]
        
        return returns, advantages
    
    def update(self):
        """
        更新策略
        """
        if len(self.buffer.states) == 0:
            return
        
        # 确保在训练模式下更新（BatchNorm需要batch统计）
        self.ac_network.train()
        
        # 获取所有数据
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get_all()
        
        # 计算最后一个状态的价值
        # 单样本推理时使用eval模式，避免BatchNorm报错
        last_state = states[-1]
        last_state_tensor = torch.tensor(last_state, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.ac_network.eval()
        with torch.no_grad():
            _, next_value = self.ac_network(last_state_tensor)
            next_value = next_value.item()
        self.ac_network.train()  # 恢复训练模式
        
        # 计算returns和advantages
        returns, advantages = self.compute_returns(rewards, values, dones, next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为tensor
        states_tensor = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        returns_tensor = torch.tensor(returns, device=self.device, dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        
        # 前向传播
        action_probs, values_pred = self.ac_network(states_tensor)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy().mean()
        
        # 计算损失
        policy_loss = -(log_probs * advantages_tensor).mean()
        value_loss = nn.MSELoss()(values_pred.squeeze(), returns_tensor)
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
        }, path + 'a2c_checkpoint.pth')
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path + 'a2c_checkpoint.pth', map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

