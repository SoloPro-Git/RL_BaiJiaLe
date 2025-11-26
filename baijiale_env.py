#! python
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BAIJIALE(gym.Env):
    """
    下注闲家而闲赢者，赢1赔1。
    下注和局（即最终点数一样者），赢1赔8。
    下注庄对子或闲对子（即庄或闲首两张牌为同数字或英文字母者），赢1赔5.5。
    
    训练目标：
    模型需要从剩余牌型中学习计算各投注方案的期望收益，
    选择期望收益最大的投注方案，如果所有投注期望<=0则不投注。
    """

    def __init__(self, money, max_steps=1280) -> None:
        """
        初始化百家乐环境
        
        Args:
            money: 初始金钱
            max_steps: 最大步数限制，默认1280。设置为None表示无限制
        """
        super().__init__()
        # 初始化卡组 卡组数量为3
        self.money = money
        self.max_steps = max_steps
        self.time_punish = False
        self.deck_num = 3  # 3副牌
        self.action_dict = {0: '不玩', 1: '庄', 2: '闲', 3: '和', 4: '对'}
        self.odds = {'不玩': 0, '庄': 1, '闲': 1, '和': 8, '对': 5.5}
        
        # 定义观察空间：52张牌，每张牌的数量（0-3）
        self.observation_space = spaces.Box(
            low=0, 
            high=3, 
            shape=(52,), 
            dtype=np.int32
        )
        
        # 定义动作空间：5个动作（不玩、庄、闲、和、对）
        self.action_space = spaces.Discrete(5)
        
        # 初始化随机数生成器
        self.np_random = None
        self.seed()

    def init_deck(self) -> list:
        self.deck = []
        self.card_names = []
        for flower in ['A', 'B', 'C', 'D']:
            for num in range(1, 14):
                card_name = flower + '_' + str(num)
                self.deck.extend([card_name] * self.deck_num)
                self.card_names.append(card_name)

    def deck_list_2_training_state(self, deck):
        state = []
        for card_name in self.card_names:
            if card_name in self.deck:
                card_number = self.deck.count(card_name)
            else:
                card_number = 0
            state.append(card_number)
        return state

    def reset(self, seed=None, options=None):
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 可选参数
            
        Returns:
            observation: 初始观察状态
            info: 包含额外信息的字典
        """
        if seed is not None:
            self.seed(seed)
        
        self.init_deck()
        self.current_money = self.money
        self.step_count = 0
        state = self.deck_list_2_training_state(self.deck)
        observation = np.array(state, dtype=np.int32)
        
        info = {
            'current_money': self.current_money,
            'step_count': self.step_count
        }
        
        return observation, info

    def sample_deck(self):
        card_a = self.deck.pop(self.np_random.integers(0, len(self.deck)))
        card_b = self.deck.pop(self.np_random.integers(0, len(self.deck)))
        return [card_a, card_b]

    def get_score(self, card: list):
        card1, card2 = card[0], card[1]
        card1 = int(card1.split('_')[-1])
        card2 = int(card2.split('_')[-1])

        card1 = card1 if card1 < 10 else 0
        card2 = card2 if card2 < 10 else 0
        return card1 + card2

    def compare_banker_player(self, banker_card, player_card):
        banker_score = self.get_score(banker_card)
        player_score = self.get_score(player_card)
        if banker_score > player_score:
            return '庄'
        if banker_score < player_score:
            return '闲'
        return '和'

    def is_pair(self, card):
        card1, card2 = card[0], card[1]
        card1 = int(card1.split('_')[-1])
        card2 = int(card2.split('_')[-1])
        return card1 == card2

    def get_reward(self, action, banker_card, player_card):
        """
        计算奖励
        
        环境只给出奖励值，不提供任何概率信息。
        模型需要从剩余牌型（状态）和奖励历史中学习计算各动作的期望收益。
        
        奖励规则：
        - 投注庄/闲：猜对+1，猜错-1
        - 投注和：猜对+8，猜错-1
        - 投注对子：中+5.5，不中-1
        - 不玩：0
        
        模型目标：从剩余牌型中学习预测各动作的期望收益，
        选择期望收益最大的动作，如果所有投注期望<=0则不投注。
        """
        action = self.action_dict[action]
        if action in ['庄', '闲', '和']:
            winner = self.compare_banker_player(banker_card, player_card)
            if action == winner:
                reward = self.odds[action]
            else:
                reward = -1
        elif action == '对':
            if self.is_pair(banker_card) or self.is_pair(player_card):
                reward = 5.5
            else:
                reward = -1
        elif action == '不玩':
            reward = 0
        return reward

    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 动作（0-4）
            
        Returns:
            observation: 新的观察状态
            reward: 奖励值
            terminated: 是否自然终止（钱输光或达到2倍初始金额）
            truncated: 是否被截断（达到最大步数）
            info: 包含额外信息的字典
        """
        if len(self.deck) < 4:
            self.init_deck()

        banker_card = self.sample_deck()
        player_card = self.sample_deck()
        next_state = self.deck_list_2_training_state(self.deck)
        reward = self.get_reward(action, banker_card, player_card)

        self.current_money += reward
        self.step_count += 1
        
        # 检查是否自然终止（钱输光或达到2倍初始金额）
        terminated = False
        if self.current_money <= 0 or self.current_money >= 2 * self.money:
            terminated = True
        
        # 检查是否达到最大步数限制
        truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps:
            truncated = True

        observation = np.array(next_state, dtype=np.int32)
        info = {
            'current_money': self.current_money,
            'step_count': self.step_count,
            'banker_card': banker_card,
            'player_card': player_card
        }

        return observation, reward, terminated, truncated, info
    
    def seed(self, seed=None):
        """
        设置随机种子
        
        Args:
            seed: 随机种子
            
        Returns:
            seed: 返回使用的种子
        """
        self.np_random = np.random.default_rng(seed)
        return [seed]
    
    def render(self):
        """
        渲染环境（可选实现）
        """
        print(f"当前金钱: {self.current_money}, 步数: {self.step_count}, 剩余牌数: {len(self.deck)}")
    
    def close(self):
        """
        清理资源（可选实现）
        """
        pass


if __name__ == "__main__":
    money = 100
    env = BAIJIALE(money, max_steps=1280)
    observation, info = env.reset()
    print(f"初始状态: 金钱={info['current_money']}, 步数={info['step_count']}")
    
    for i_step in range(10000):
        action = 4  # 选择动作"对"
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"步数 {i_step}: 奖励={reward}, 金钱={info['current_money']}, 步数计数={info['step_count']}")
        
        if terminated or truncated:
            print(f"回合结束: terminated={terminated}, truncated={truncated}")
            break
