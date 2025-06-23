# train.py (MODIFIED to use per-step reward metric)
import os, torch, numpy as np
from collections import deque, defaultdict
from sumo_env import SumoMultiAgentEnv
from ppo_agent import PPO
import matplotlib.pyplot as plt

# ================== [MODIFIED] 配置与超参数 ==================
SUMOCFG_FILE = "highway.sumocfg" 
VEHICLE_PREFIXES = ['main_', 'ramp_']
USE_GUI = False
MAX_EPISODE_STEPS = 1000
MAX_TRAINING_TIMESTEPS = 40000 # 建议增加总训练步数以观察长期效果

# --- 核心训练超参数调整 ---
UPDATE_TIMESTEP = 4096      # 更频繁地更新策略 (原: 8192)
BATCH_SIZE = 256            # 相应减小Batch Size (原: 512)
K_EPOCHS = 4                # 减少每个batch的训练次数，防止过拟合 (原: 10)
GAMMA = 0.99
EPS_CLIP = 0.2
LR_ACTOR = 0.0003
LR_CRITIC = 0.001

# --- 探索策略参数调整 ---
ACTION_STD_INIT = 0.4               # 稍小的初始探索标准差 (原: 0.5)
ACTION_STD_DECAY_RATE = 0.05        # [NEW] 动作标准差衰减率
MIN_ACTION_STD = 0.1                # [NEW] 最小动作标准差

MODEL_NAME = "MARL_PPO_SUMO_Optimized"
SAVE_MODEL_FREQ = UPDATE_TIMESTEP * 5 # 按更新次数的整数倍保存
ANALYZE_FREQ_EPISODES = 2 # 增加分析频率以获得更平滑的曲线
PATIENCE = 20
MIN_DELTA = 0.01 


# ================== 绘图函数 (已修改以适配新的Y轴标签) ==================

def plot_learning_curve(episodes, rewards, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, color='blue', marker='o', linestyle='-')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Episode')
    # [MODIFIED] Y轴标签反映新的度量标准
    plt.ylabel(f'Avg Reward per Step (per {ANALYZE_FREQ_EPISODES} episodes)')
    plt.grid(True)
    plt.savefig(f'{model_name}_learning_curve.png')
    print(f"--- 学习曲线图已保存为: {model_name}_learning_curve.png ---")

def plot_sub_rewards_curve(episodes, sub_rewards_data, model_name):
    plt.figure(figsize=(12, 8))
    for name, values in sub_rewards_data.items():
        plt.plot(episodes, values, marker='o', linestyle='-', label=name)
    
    plt.title(f'Sub-Rewards Analysis - {model_name}')
    plt.xlabel('Episode')
    # [MODIFIED] Y轴标签反映新的度量标准
    plt.ylabel(f'Avg Sub-Reward per Step (per {ANALYZE_FREQ_EPISODES} episodes)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{model_name}_sub_rewards_curve.png')
    print(f"--- 子奖励曲线图已保存为: {model_name}_sub_rewards_curve.png ---")


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = SumoMultiAgentEnv(SUMOCFG_FILE, vehicle_prefixes=VEHICLE_PREFIXES, use_gui=USE_GUI, max_episode_steps=MAX_EPISODE_STEPS)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化PPO agent时传入初始动作标准差
    ppo_agent = PPO(state_dim, action_dim, LR_ACTOR, LR_CRITIC, GAMMA, K_EPOCHS, EPS_CLIP, BATCH_SIZE, ACTION_STD_INIT)

    print(f"开始多智能体训练, 设备: {device}")
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    total_simulation_steps, i_episode, last_save_sim_step = 0, 0, 0
    best_avg_reward, patience_counter = -np.inf, 0
    
    ep_rewards_deque = deque(maxlen=ANALYZE_FREQ_EPISODES)
    sub_rewards_deques = defaultdict(lambda: deque(maxlen=ANALYZE_FREQ_EPISODES))

    plot_episodes, plot_avg_rewards = [], []
    plot_sub_rewards_data = defaultdict(list)

    while total_simulation_steps < MAX_TRAINING_TIMESTEPS:
        states = env.reset()
        current_ep_total_reward = 0
        ep_total_sub_rewards = defaultdict(float)
        agent_context = {}
        i_episode += 1

        # [NEW] 学习率与探索率退火
        # 计算当前训练进度比例
        frac = 1.0 - (total_simulation_steps / MAX_TRAINING_TIMESTEPS)
        
        # 线性衰减学习率
        new_lr = LR_ACTOR * frac
        ppo_agent.optimizer.param_groups[0]['lr'] = new_lr
        ppo_agent.optimizer.param_groups[1]['lr'] = LR_CRITIC * frac # Critic也衰减

        # 指数衰减动作标准差 (在PPO Agent内部实现)
        current_action_std = ppo_agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

        for t in range(1, MAX_EPISODE_STEPS + 1):
            if not states: break
                
            actions = {}
            for agent_id, state in states.items():
                action, log_prob, value = ppo_agent.select_action(state)
                actions[agent_id] = action
                agent_context[agent_id] = (state, log_prob, value)

            next_states, rewards, dones, infos = env.step(actions)
            total_simulation_steps += 1

            for agent_id, action in actions.items():
                if agent_id in rewards and agent_id in agent_context:
                    state, log_prob, value = agent_context[agent_id]
                    reward = rewards[agent_id]
                    done = dones.get(agent_id, False)
                    ppo_agent.memory.store(state, action, log_prob, reward, done, value)
                    
            current_ep_total_reward += sum(rewards.values())
            
            for agent_id, info_dict in infos.items():
                if info_dict:
                    # [MODIFIED] 确保所有子奖励都被记录
                    for name, val in info_dict.items():
                         ep_total_sub_rewards[name] += val
            
            if len(ppo_agent.memory.states) >= UPDATE_TIMESTEP:
                ppo_agent.update()
                # [MODIFIED] 更新后立即衰减下一次的探索率
                # ppo_agent.decay_action_std(ACTION_STD_DECAY_RATE, MIN_ACTION_STD)

            if total_simulation_steps - last_save_sim_step >= SAVE_MODEL_FREQ:
                ppo_agent.save(f"{MODEL_NAME}.pth")
                last_save_sim_step = total_simulation_steps

            states = next_states
            if dones.get('__all__', False): 
                break
        
        num_steps_in_ep = t
        avg_reward_per_step = current_ep_total_reward / num_steps_in_ep if num_steps_in_ep > 0 else 0
        ep_rewards_deque.append(avg_reward_per_step)
        
        for name, total_val in ep_total_sub_rewards.items():
            avg_sub_reward_per_step = total_val / (num_steps_in_ep * len(infos)) if (num_steps_in_ep * len(infos)) > 0 else 0
            sub_rewards_deques[name].append(avg_sub_reward_per_step)

        if i_episode % ANALYZE_FREQ_EPISODES == 0 and len(ep_rewards_deque) > 0:
            avg_reward_in_window = np.mean(ep_rewards_deque)
            
            plot_episodes.append(i_episode)
            plot_avg_rewards.append(avg_reward_in_window)
            for name, d in sub_rewards_deques.items():
                plot_sub_rewards_data[name].append(np.mean(d))

            print(f"Episode: {i_episode:<5} | SimSteps: {total_simulation_steps:<7} | Avg Reward/Step: {avg_reward_in_window:.4f} | LR: {new_lr:.6f} | Action Std: {current_action_std:.4f}")
            
            if avg_reward_in_window > best_avg_reward + MIN_DELTA:
                print(f"  >> New best model! Avg reward from {best_avg_reward:.4f} to {avg_reward_in_window:.4f}. Saving model.")
                best_avg_reward = avg_reward_in_window
                ppo_agent.save(f"{MODEL_NAME}_best.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                print("! Early stopping due to no improvement. !")
                break
        
        if patience_counter >= PATIENCE: break

    env.close()
    
    if plot_episodes:
        plot_learning_curve(plot_episodes, plot_avg_rewards, MODEL_NAME)
        plot_sub_rewards_curve(plot_episodes, plot_sub_rewards_data, MODEL_NAME)
        
    print("\n================== TRAINING COMPLETE ==================")

if __name__ == '__main__':
    if not os.path.exists(SUMOCFG_FILE):
        print(f"错误: 配置文件未找到 '{SUMOCFG_FILE}'。请确保路径正确。")
    else:
        train()