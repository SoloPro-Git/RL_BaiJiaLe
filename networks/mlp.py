#!/usr/bin/env python
# coding=utf-8
"""
通用的多层感知机（MLP）网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    通用的多层感知机，用于值函数估计（如DQN）
    """
    
    def __init__(self, n_states, n_actions, hidden_dims=[512, 512, 512]):
        """
        初始化MLP网络
        
        Args:
            n_states: 输入状态维度
            n_actions: 输出动作维度
            hidden_dims: 隐藏层维度列表
        """
        super(MLP, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        
        layers = []
        input_dim = n_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ValueNetwork(nn.Module):
    """
    价值网络，用于估计状态价值 V(s)
    """
    
    def __init__(self, n_states, hidden_dims=[512, 512, 512]):
        """
        初始化价值网络
        
        Args:
            n_states: 输入状态维度
            hidden_dims: 隐藏层维度列表
        """
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        
        layers = []
        input_dim = n_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    """
    策略网络，用于输出动作概率分布 π(a|s)
    """
    
    def __init__(self, n_states, n_actions, hidden_dims=[512, 512, 512]):
        """
        初始化策略网络
        
        Args:
            n_states: 输入状态维度
            n_actions: 输出动作维度
            hidden_dims: 隐藏层维度列表
        """
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        
        layers = []
        input_dim = n_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, n_actions)
    
    def forward(self, x):
        x = self.shared_layers(x)
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs
    
    def get_action_and_log_prob(self, x):
        """
        获取动作和对应的对数概率
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
        """
        x = self.shared_layers(x)
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ActorCritic(nn.Module):
    """
    Actor-Critic 网络，同时输出策略和价值
    """
    
    def __init__(self, n_states, n_actions, hidden_dims=[512, 512, 512]):
        """
        初始化Actor-Critic网络
        
        Args:
            n_states: 输入状态维度
            n_actions: 输出动作维度
            hidden_dims: 隐藏层维度列表
        """
        super(ActorCritic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        
        # 共享的特征提取层
        layers = []
        input_dim = n_states
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor 头（策略网络）
        self.actor_head = nn.Linear(input_dim, n_actions)
        
        # Critic 头（价值网络）
        self.critic_head = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Returns:
            action_probs: 动作概率分布
            value: 状态价值
        """
        x = self.shared_layers(x)
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic_head(x)
        return action_probs, value
    
    def get_action_and_value(self, x):
        """
        获取动作、对数概率和价值
        
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        x = self.shared_layers(x)
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_head(x)
        return action, log_prob, value

