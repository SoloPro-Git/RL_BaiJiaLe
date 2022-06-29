#! python
from random import randint
import re


class BAIJIALE:
    """
    下注闲家而闲赢者，赢1赔1。
    下注和局（即最终点数一样者），赢1赔8。
    下注庄对子或闲对子（即庄或闲首两张牌为同数字或英文字母者），赢1赔5.5。
    """

    def __init__(self, money) -> None:
        # 初始化卡组 卡组数量为3
        self.money = money
        self.time_punish = False
        self.deck_num = 3
        self.action_dict = {0: '不玩', 1: '庄', 2: '闲', 3: '和', 4: '对'}
        self.odds = {'不玩': 0, '庄': 1, '闲': 1, '和': 8, '对': 5.5}
        self.observation_space = 13 * 4

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

    def reset(self) -> list:
        self.init_deck()
        self.current_money = self.money
        state = self.deck_list_2_training_state(self.deck)
        return state

    def sample_deck(self):
        card_a = self.deck.pop(randint(0, len(self.deck)-1))
        card_b = self.deck.pop(randint(0, len(self.deck)-1))
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
        if len(self.deck) < 4:
            self.init_deck()

        done = False
        banker_card = self.sample_deck()
        player_card = self.sample_deck()
        next_state = self.deck_list_2_training_state(self.deck)
        reward = self.get_reward(action, banker_card, player_card)

        self.current_money += reward
        if self.current_money <= 0 or self.current_money >= 2 * self.money:
            done = True

        return next_state, reward, done, self.current_money


if __name__ == "__main__":
    money = 100
    env = BAIJIALE(money)
    MAX_STEPS = 10000
    state = env.reset()
    for i_step in range(MAX_STEPS):
        action = 4
        next_state, reward, done, cur_money = env.step(action)
        print(i_step, reward, cur_money)
        state = next_state
        if done:
            break
