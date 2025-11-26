#!/usr/bin/env python
# coding=utf-8
"""
经验生成工具
用于生成预填充经验回放缓冲区的专家经验
"""

import numpy as np
import random
from collections import Counter
from baijiale_env import BAIJIALE


def get_card_value(card_name):
    """获取牌的点数（10-K计为0）"""
    num = int(card_name.split('_')[-1])
    return num if num < 10 else 0


def calculate_pair_probability(deck):
    """
    计算对子的概率
    
    Args:
        deck: 剩余牌堆
        
    Returns:
        prob: 对子概率
    """
    if len(deck) < 4:
        return 0.0
    
    total_pairs = 0
    total_combinations = 0
    
    # 计算庄对子和闲对子的概率
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            card1_num = int(deck[i].split('_')[-1])
            card2_num = int(deck[j].split('_')[-1])
            if card1_num == card2_num:
                total_pairs += 1
            total_combinations += 1
    
    # 简化计算：从剩余牌中随机选4张，计算对子概率
    if len(deck) < 4:
        return 0.0
    
    # 使用蒙特卡洛方法估算
    pair_count = 0
    n_samples = min(1000, len(deck) * 10)
    for _ in range(n_samples):
        if len(deck) >= 4:
            sampled = random.sample(deck, 4)
            banker_pair = int(sampled[0].split('_')[-1]) == int(sampled[1].split('_')[-1])
            player_pair = int(sampled[2].split('_')[-1]) == int(sampled[3].split('_')[-1])
            if banker_pair or player_pair:
                pair_count += 1
    
    return pair_count / n_samples if n_samples > 0 else 0.0


def calculate_expected_value_mc(env, state, action, n_simulations=1000):
    """
    使用蒙特卡洛方法计算状态-动作的期望收益
    
    Args:
        env: 环境实例
        state: 当前状态（52维向量）
        action: 动作
        n_simulations: 模拟次数
        
    Returns:
        expected_value: 期望收益
    """
    # 从状态重建牌堆
    deck = []
    for i, card_name in enumerate(env.card_names):
        count = int(state[i])
        deck.extend([card_name] * count)
    
    if len(deck) < 4:
        return -1.0  # 牌不够，期望为负
    
    total_reward = 0.0
    valid_samples = 0
    
    for _ in range(n_simulations):
        if len(deck) < 4:
            break
        
        # 复制牌堆
        test_deck = deck.copy()
        
        # 随机抽取4张牌
        if len(test_deck) < 4:
            continue
        
        try:
            banker_card = [test_deck.pop(random.randrange(len(test_deck))),
                          test_deck.pop(random.randrange(len(test_deck)))]
            player_card = [test_deck.pop(random.randrange(len(test_deck))),
                          test_deck.pop(random.randrange(len(test_deck)))]
            
            reward = env.get_reward(action, banker_card, player_card)
            total_reward += reward
            valid_samples += 1
        except (IndexError, ValueError):
            continue
    
    return total_reward / valid_samples if valid_samples > 0 else -1.0


def find_best_action(env, state, n_simulations=500):
    """
    找到当前状态下的最优动作
    
    Args:
        env: 环境实例
        state: 当前状态
        n_simulations: 每个动作的模拟次数
        
    Returns:
        best_action: 最优动作
        expected_values: 各动作的期望收益
    """
    expected_values = []
    
    for action in range(5):  # 5个动作
        if action == 0:  # 不玩
            expected_values.append(0.0)
        else:
            ev = calculate_expected_value_mc(env, state, action, n_simulations)
            expected_values.append(ev)
    
    # 选择期望收益最大的动作
    best_action = np.argmax(expected_values)
    
    # 如果所有投注的期望都<=0，选择不玩
    if best_action > 0 and expected_values[best_action] <= 0:
        best_action = 0
    
    return best_action, expected_values


def generate_special_states(env, n_states=1000):
    """
    生成特殊牌型状态
    
    Args:
        env: 环境实例
        n_states: 生成状态数量
        
    Returns:
        states: 状态列表
    """
    states = []
    
    for _ in range(n_states):
        # 初始化牌堆
        env.init_deck()
        deck = env.deck.copy()
        
        # 随机移除一些牌，创建特殊状态
        remove_count = random.randint(0, min(50, len(deck) - 4))
        
        for _ in range(remove_count):
            if len(deck) > 4:
                deck.pop(random.randrange(len(deck)))
        
        # 转换为状态向量
        state = env.deck_list_2_training_state(deck)
        states.append(state)
    
    return states


def generate_expert_experiences(env, n_experiences=2000, n_simulations=500):
    """
    生成专家经验
    
    Args:
        env: 环境实例
        n_experiences: 生成经验数量
        n_simulations: 每个状态-动作对的模拟次数
        
    Returns:
        experiences: 经验列表，每个元素为 (state, action, reward, next_state, terminated, truncated)
    """
    experiences = []
    
    print(f"开始生成 {n_experiences} 个专家经验...")
    
    # 生成特殊状态
    special_states = generate_special_states(env, n_experiences)
    
    for i, state in enumerate(special_states):
        if (i + 1) % 200 == 0:
            print(f"已生成 {i + 1}/{n_experiences} 个经验...")
        
        # 找到最优动作
        best_action, expected_values = find_best_action(env, state, n_simulations)
        
        # 如果最优动作的期望收益<=0，跳过（不生成不玩的经验，因为价值不大）
        if best_action == 0 or expected_values[best_action] <= 0:
            continue
        
        # 从状态重建牌堆
        deck = []
        for j, card_name in enumerate(env.card_names):
            count = int(state[j])
            deck.extend([card_name] * count)
        
        if len(deck) < 4:
            continue
        
        # 模拟执行动作，生成经验
        try:
            # 抽取4张牌
            banker_card = [deck.pop(random.randrange(len(deck))),
                          deck.pop(random.randrange(len(deck)))]
            player_card = [deck.pop(random.randrange(len(deck))),
                          deck.pop(random.randrange(len(deck)))]
            
            # 计算奖励
            reward = env.get_reward(best_action, banker_card, player_card)
            
            # 获取下一状态
            next_state = env.deck_list_2_training_state(deck)
            
            # 判断是否终止（简化处理，这里不判断金钱）
            terminated = False
            truncated = False
            
            # 如果牌不够了，标记为截断
            if len(deck) < 4:
                truncated = True
            
            experiences.append((state, best_action, reward, next_state, terminated, truncated))
            
        except (IndexError, ValueError):
            continue
    
    print(f"成功生成 {len(experiences)} 个专家经验")
    return experiences

