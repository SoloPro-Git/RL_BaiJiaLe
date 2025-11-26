#!/usr/bin/env python
# coding=utf-8
"""
统一的强化学习训练入口
支持多种算法：DQN, PPO, A2C, REINFORCE
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from baijiale_env import BAIJIALE
from agents import DQNAgent, PPOAgent, A2CAgent, REINFORCEAgent
from configs.config import get_config, get_agent_config, create_output_paths
from utils import plot_rewards_cn, save_results, make_dir


def create_agent(algo_name, state_dim, action_dim, device, agent_config, env=None):
    """
    创建Agent实例
    
    Args:
        algo_name: 算法名称
        state_dim: 状态维度
        action_dim: 动作维度
        device: 计算设备
        agent_config: Agent配置
        env: 环境实例（用于预填充经验，所有算法都需要）
        
    Returns:
        agent: Agent实例
    """
    if algo_name.lower() == 'dqn':
        agent = DQNAgent(state_dim, action_dim, device, agent_config, env=env)
    elif algo_name.lower() == 'ppo':
        agent = PPOAgent(state_dim, action_dim, device, agent_config, env=env)
    elif algo_name.lower() == 'a2c':
        agent = A2CAgent(state_dim, action_dim, device, agent_config, env=env)
    elif algo_name.lower() == 'reinforce':
        agent = REINFORCEAgent(state_dim, action_dim, device, agent_config, env=env)
    else:
        raise ValueError(f"未知的算法名称: {algo_name}")
    
    return agent


def train(env, agent, config, algo_name):
    """
    训练函数
    
    Args:
        env: 环境实例
        agent: Agent实例
        config: 配置字典
        algo_name: 算法名称
        
    Returns:
        rewards: 奖励列表
        ma_rewards: 滑动平均奖励列表
    """
    print('=' * 50)
    print(f'开始训练!')
    print(f'环境：{config["env_name"]}')
    print(f'算法：{algo_name.upper()}')
    print(f'设备：{config["device"]}')
    print(f'训练回合数：{config["train_eps"]}')
    print('=' * 50)
    
    rewards = []
    ma_rewards = []
    agent.train()
    
    for i_ep in range(config['train_eps']):
        ep_reward = 0
        observation, info = env.reset()
        step_count = 0
        
        # 根据算法类型初始化缓冲区
        if algo_name.lower() in ['ppo', 'a2c', 'reinforce']:
            # On-policy算法需要收集完整episode
            pass
        
        while True:
            # 选择动作
            if algo_name.lower() == 'dqn':
                action = agent.select_action(observation, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储到经验回放（非保护经验）
                agent.memory.push(observation, action, reward, next_obs, terminated, truncated, protected=False)
                
                # 更新Agent
                agent.update(times=1)
                
            elif algo_name.lower() == 'ppo':
                action, log_prob, value = agent.select_action(observation, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储到缓冲区
                agent.buffer.push(observation, action, reward, log_prob, value, done)
                
            elif algo_name.lower() == 'a2c':
                action, log_prob, value = agent.select_action(observation, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储到缓冲区
                agent.buffer.push(observation, action, reward, log_prob, value, done)
                
            elif algo_name.lower() == 'reinforce':
                action, log_prob = agent.select_action(observation, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储到缓冲区
                agent.buffer.push(observation, action, reward, log_prob, done)
            
            observation = next_obs
            ep_reward += reward
            step_count += 1
            
            if done:
                break
        
        # On-policy算法在episode结束后更新
        if algo_name.lower() in ['ppo', 'a2c', 'reinforce']:
            agent.update()
        
        # 记录奖励
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        
        # 打印进度
        if (i_ep + 1) % 10 == 0:
            print(f'回合：{i_ep+1}/{config["train_eps"]}, '
                  f'奖励：{ep_reward:.2f}, '
                  f'平均奖励：{np.mean(rewards[-10:]):.2f}, '
                  f'剩余金钱：{info["current_money"]:.2f}, '
                  f'步数：{info["step_count"]}')
    
    print('完成训练！')
    return rewards, ma_rewards


def test(env, agent, config, algo_name):
    """
    测试函数
    
    Args:
        env: 环境实例
        agent: Agent实例
        config: 配置字典
        algo_name: 算法名称
        
    Returns:
        rewards: 奖励列表
        ma_rewards: 滑动平均奖励列表
    """
    print('=' * 50)
    print(f'开始测试!')
    print(f'环境：{config["env_name"]}')
    print(f'算法：{algo_name.upper()}')
    print(f'设备：{config["device"]}')
    print(f'测试回合数：{config["test_eps"]}')
    print('=' * 50)
    
    rewards = []
    ma_rewards = []
    agent.eval()
    
    for i_ep in range(config['test_eps']):
        ep_reward = 0
        observation, info = env.reset()
        
        while True:
            action = agent.select_action(observation, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observation = next_obs
            ep_reward += reward
            
            if done:
                break
        
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        
        print(f"回合：{i_ep+1}/{config['test_eps']}，"
              f"奖励：{ep_reward:.2f}, "
              f"剩余金钱：{info['current_money']:.2f}, "
              f"步数：{info['step_count']}")
    
    print('完成测试！')
    return rewards, ma_rewards


def main():
    parser = argparse.ArgumentParser(description='强化学习训练入口')
    parser.add_argument('--algo', type=str, default='dqn',
                       choices=['dqn', 'ppo', 'a2c', 'reinforce'],
                       help='选择算法 (default: dqn)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='训练或测试模式 (default: train)')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练回合数（覆盖配置）')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='环境最大步数（None表示无限制）')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型加载路径（测试模式必需）')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_config()
    agent_config = get_agent_config(args.algo)
    
    # 覆盖配置
    if args.episodes is not None:
        config['train_eps'] = args.episodes
    if args.max_steps is not None:
        config['max_steps'] = args.max_steps
    
    # 创建环境
    env = BAIJIALE(config['init_money'], max_steps=config['max_steps'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 创建Agent（所有算法都需要env用于预填充经验）
    agent = create_agent(args.algo, state_dim, action_dim, config['device'], agent_config, env=env)
    
    # 创建输出路径
    result_path, model_path = create_output_paths(config['env_name'], args.algo)
    config['result_path'] = result_path
    config['model_path'] = model_path
    
    # 创建绘图配置
    class PlotConfig:
        def __init__(self):
            self.env_name = config['env_name']
            self.algo_name = args.algo.upper()
            self.device = config['device']
            self.result_path = result_path
            self.model_path = model_path
            self.save = True
    
    plot_cfg = PlotConfig()
    
    if args.mode == 'train':
        # 训练
        rewards, ma_rewards = train(env, agent, config, args.algo)
        
        # 保存模型
        agent.save(model_path)
        print(f'模型已保存到: {model_path}')
        
        # 保存结果
        save_results(rewards, ma_rewards, tag='train', path=result_path)
        
        # 绘制曲线
        plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train')
        print(f'结果已保存到: {result_path}')
        
    elif args.mode == 'test':
        # 测试
        if args.model_path is None:
            raise ValueError("测试模式需要指定 --model_path 参数")
        
        # 加载模型
        agent.load(args.model_path)
        print(f'模型已从 {args.model_path} 加载')
        
        # 测试
        rewards, ma_rewards = test(env, agent, config, args.algo)
        
        # 保存结果
        save_results(rewards, ma_rewards, tag='test', path=result_path)
        
        # 绘制曲线
        plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='test')
        print(f'结果已保存到: {result_path}')


if __name__ == "__main__":
    main()
