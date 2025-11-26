#!/usr/bin/env python
# coding=utf-8
"""
统一配置管理
"""

import torch
import datetime
import os


def _get_device():
    """
    自动检测并返回最佳计算设备
    
    Returns:
        device: torch.device 对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) 芯片的 MPS 后端
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_config():
    """
    获取通用配置
    
    Returns:
        config: 配置字典
    """
    config = {
        # 环境配置
        'env_name': 'BAIJIALE-v1',
        'init_money': 100,
        'max_steps': 1280,  # None表示无限制
        
        # 训练配置
        'train_eps': 10000,
        'test_eps': 20,
        
        # 设备配置（优先使用 CUDA，然后是 MPS（Apple Silicon），最后是 CPU）
        'device': _get_device(),
        
        # 结果保存路径
        'result_path': './outputs/',
        'model_path': './outputs/',
    }
    return config


def get_agent_config(algo_name):
    """
    获取特定算法的配置
    
    Args:
        algo_name: 算法名称 (dqn/ppo/a2c/reinforce)
        
    Returns:
        config: 算法配置字典
    """
    base_config = {
        'gamma': 0.99,
        'hidden_dims': [512, 512, 512],
    }
    
    if algo_name.lower() == 'dqn':
        config = {
            **base_config,
            'epsilon_start': 0.99,
            'epsilon_end': 0.005,
            'epsilon_decay': 500,
            'lr': 0.0001,
            'memory_capacity': 10000,
            'batch_size': 128,
            'target_update': 4,
        }
    elif algo_name.lower() == 'ppo':
        config = {
            **base_config,
            'gae_lambda': 0.95,
            'lr': 0.0003,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'update_epochs': 4,
        }
    elif algo_name.lower() == 'a2c':
        config = {
            **base_config,
            'lr': 0.0003,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'n_steps': 5,
        }
    elif algo_name.lower() == 'reinforce':
        config = {
            **base_config,
            'lr': 0.0003,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
        }
    else:
        raise ValueError(f"未知的算法名称: {algo_name}")
    
    return config


def create_output_paths(env_name, algo_name):
    """
    创建输出路径
    
    Args:
        env_name: 环境名称
        algo_name: 算法名称
        
    Returns:
        result_path: 结果保存路径
        model_path: 模型保存路径
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_path = f"./outputs/{env_name}/{algo_name}/{curr_time}/"
    result_path = base_path + "results/"
    model_path = base_path + "models/"
    
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    return result_path, model_path

