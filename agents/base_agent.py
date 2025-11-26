#!/usr/bin/env python
# coding=utf-8
"""
Agent 基类接口定义
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    所有强化学习 Agent 的基类
    定义了统一的接口规范
    """
    
    def __init__(self, state_dim, action_dim, device='cpu'):
        """
        初始化 Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            device: 计算设备（'cpu' 或 'cuda'）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.training = True
    
    @abstractmethod
    def select_action(self, state, training=True):
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            training: 是否处于训练模式
            
        Returns:
            action: 选择的动作
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """
        更新策略/值函数
        不同算法的更新方式不同，参数可以灵活定义
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        raise NotImplementedError
    
    def train(self):
        """设置为训练模式"""
        self.training = True
    
    def eval(self):
        """设置为评估模式"""
        self.training = False

