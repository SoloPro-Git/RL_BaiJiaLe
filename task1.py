#!/usr/bin/env python
# coding=utf-8

from dqn import DQN
from utils import plot_rewards, plot_rewards_cn
from utils import save_results, make_dir
import torch.nn.functional as F
import torch.nn as nn
import datetime
import torch
from baijiale_env import BAIJIALE
import sys
import os
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = "DQN"  # 算法名称
env_name = 'BAIJIALE-v1'  # 环境名称


class DQNConfig:
    ''' 算法相关参数设置
    '''

    def __init__(self):
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.train_eps = 1000  # 训练的回合数
        self.test_eps = 20  # 测试的回合数
        # 超参数
        self.gamma = 0.99  # 强化学习中的折扣因子
        self.epsilon_start = 0.99  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.005  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 12800  # 经验回放的容量
        self.batch_size = 128  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.hidden_dim = 512  # 网络隐藏层


class PlotConfig:
    ''' 绘图相关参数设置
    '''

    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            n_states: 输入的特征数即环境的状态维度
            n_actions: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc2_2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, n_actions)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_2(x))
        return self.fc3(x)


def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    init_money = 100
    env = BAIJIALE(init_money)  # 创建环境  
    n_states = env.observation_space  # 状态维度
    n_actions = len(env.action_dict)  # 动作维度
    model = MLP(n_states, n_actions)
    agent = DQN(n_actions, model, cfg)  # 创建智能体
    return env, agent

def init_action_count(agent):
    action_count = {}
    for i in range(agent.n_actions):
        action_count[i] = 0
    return action_count

def train(cfg, env, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        max_ep_try = cfg.batch_size * 10 # policy_net 的尝试次数
        ep_try = 0
        action_count = init_action_count(agent)
        while ep_try <= max_ep_try:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, cur_money = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            # print(action,state,reward,cur_money)
            ep_try += 1
            action_count[action] += 1
            if done:
                break
        agent.update(10*ep_try)  # 更新智能体
        if (i_ep+1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 1 == 0:
            print('回合：{}/{}, 奖励：{}, 剩余金钱：{}'.format(i_ep+1, cfg.train_eps, ep_reward, cur_money))
        if (i_ep+1) % 10 == 0: 
            print(f'回合: {i_ep+1}, 动作选择次数为{action_count}')
            make_dir(plot_cfg.result_path, plot_cfg.model_path)
            agent.save(path=plot_cfg.model_path)
    print('完成训练！')
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    # 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, cur_money = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}, 剩余金钱：{cur_money:.1f}")
    print('完成测试！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DQNConfig()
    plot_cfg = PlotConfig()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=plot_cfg.model_path)  # 保存模型
    save_results(rewards, ma_rewards, tag='train',
                 path=plot_cfg.result_path)  # 保存结果
    plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果
    # 测试


    # env, agent = env_agent_config(cfg)
    # agent.load(path=plot_cfg.model_path)  # 导入模型
    # rewards, ma_rewards = test(cfg, env, agent)
    # save_results(rewards, ma_rewards, tag='test',
    #              path=plot_cfg.result_path)  # 保存结果
    # plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag="test")  # 画出结果
