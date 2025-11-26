# 强化学习算法框架使用说明

## 支持的算法

- **DQN** (Deep Q-Network): Off-policy 值函数方法
- **PPO** (Proximal Policy Optimization): On-policy 策略梯度方法
- **A2C** (Advantage Actor-Critic): On-policy Actor-Critic方法
- **REINFORCE**: 经典策略梯度方法

## 使用方法

### 训练

```bash
# 使用DQN算法训练
python main.py --algo dqn --mode train --episodes 10000

# 使用PPO算法训练
python main.py --algo ppo --mode train --episodes 10000

# 使用A2C算法训练
python main.py --algo a2c --mode train --episodes 10000

# 使用REINFORCE算法训练
python main.py --algo reinforce --mode train --episodes 10000
```

### 测试

```bash
# 测试训练好的模型
python main.py --algo dqn --mode test --model_path ./outputs/BAIJIALE-v1/dqn/YYYYMMDD-HHMMSS/models/
```

### 参数说明

- `--algo`: 选择算法 (dqn/ppo/a2c/reinforce)
- `--mode`: 训练或测试模式 (train/test)
- `--episodes`: 训练回合数（可选，覆盖默认配置）
- `--max_steps`: 环境最大步数（可选，None表示无限制）
- `--model_path`: 模型加载路径（测试模式必需）

## 文件结构

```
RL_BaiJiaLe/
├── agents/              # Agent实现
│   ├── base_agent.py    # Agent基类
│   ├── dqn_agent.py     # DQN实现
│   ├── ppo_agent.py     # PPO实现
│   ├── a2c_agent.py     # A2C实现
│   └── reinforce_agent.py  # REINFORCE实现
├── networks/            # 神经网络结构
│   └── mlp.py          # MLP网络定义
├── configs/             # 配置管理
│   └── config.py       # 配置函数
├── baijiale_env.py     # 百家乐环境
├── main.py             # 统一入口
└── utils.py            # 工具函数
```

## 配置说明

所有算法的配置都在 `configs/config.py` 中定义，可以通过修改该文件来调整超参数。

### DQN配置
- `gamma`: 折扣因子 (0.99)
- `epsilon_start`: 初始探索率 (0.99)
- `epsilon_end`: 最终探索率 (0.005)
- `epsilon_decay`: 探索率衰减率 (500)
- `lr`: 学习率 (0.0001)
- `memory_capacity`: 经验回放容量 (10000)
- `batch_size`: 批次大小 (128)
- `target_update`: 目标网络更新频率 (4)

### PPO配置
- `gamma`: 折扣因子 (0.99)
- `gae_lambda`: GAE lambda参数 (0.95)
- `lr`: 学习率 (0.0003)
- `clip_epsilon`: PPO裁剪参数 (0.2)
- `value_coef`: 价值损失系数 (0.5)
- `entropy_coef`: 熵系数 (0.01)
- `max_grad_norm`: 梯度裁剪 (0.5)
- `update_epochs`: 每次更新的epoch数 (4)

### A2C配置
- `gamma`: 折扣因子 (0.99)
- `lr`: 学习率 (0.0003)
- `value_coef`: 价值损失系数 (0.5)
- `entropy_coef`: 熵系数 (0.01)
- `max_grad_norm`: 梯度裁剪 (0.5)
- `n_steps`: n步回报 (5)

### REINFORCE配置
- `gamma`: 折扣因子 (0.99)
- `lr`: 学习率 (0.0003)
- `entropy_coef`: 熵系数 (0.01)
- `max_grad_norm`: 梯度裁剪 (0.5)

## 输出

训练和测试的结果会保存在 `./outputs/` 目录下，包括：
- 模型文件 (`.pth`)
- 奖励曲线数据 (`.npy`)
- 奖励曲线图片 (`.png`)

