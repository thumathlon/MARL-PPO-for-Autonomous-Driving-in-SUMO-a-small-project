# evaluate.py
import os
import torch
import numpy as np
from collections import defaultdict
import traci

from sumo_env import SumoMultiAgentEnv
from ppo_agent import PPO

# --- 评估配置 ---
SUMO_CFG_FILE = "highway.sumocfg" # 确保与训练时使用的文件一致
model_path = "D:\project\RL_sumo\MARL_PPO_SUMO_Optimized_best.pth"
VEHICLE_PREFIXES = ['main_', 'ramp_']
NUM_EVAL_EPISODES = 20  # 运行多个回合以获得可靠的平均指标
MAX_EPISODE_STEPS = 1500 # 单个评估回合的最大步数
EMERGENCY_BRAKE_THRESHOLD = -4.5 # m/s^2, 定义紧急刹车的阈值

# --- Helper: Jain's Fairness Index 计算函数 ---
def jains_fairness_index(travel_times):
    """根据通行时间列表计算Jain公平性指数"""
    if not travel_times:
        return 0.0
    
    travel_times_array = np.array(travel_times)
    sum_ti = np.sum(travel_times_array)
    sum_ti_sq = np.sum(np.square(travel_times_array))
    n = len(travel_times_array)
    
    if sum_ti_sq == 0:
        return 1.0 # 如果所有时间都是0，则完全公平
        
    return (sum_ti ** 2) / (n * sum_ti_sq)


def evaluate(model_path, sumocfg_file):
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到 '{model_path}'")
        return

    # --- 1. 初始化环境和智能体 ---
    env = SumoMultiAgentEnv(sumocfg_file, vehicle_prefixes=VEHICLE_PREFIXES, use_gui=False, max_episode_steps=MAX_EPISODE_STEPS)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 我们只需要一个PPO实例来加载模型和选择动作，超参数不重要
    ppo_agent = PPO(state_dim, action_dim, 0.0003, 0.001, 0.99, 10, 0.2, 512)
    ppo_agent.load(model_path)
    print(f"--- 模型 {model_path} 加载成功 ---")

    # --- 2. 数据收集容器 ---
    # 这个列表将存储所有回合中，所有车辆的最终数据
    all_vehicle_stats = []
    total_simulation_time = 0

    print(f"--- 开始评估，共运行 {NUM_EVAL_EPISODES} 个回合 ---")
    for i_episode in range(1, NUM_EVAL_EPISODES + 1):
        states = env.reset()
        
        # 记录当前回合车辆信息的字典
        # key: vehicle_id, value: { 'start_time': float, 'emergency_brakes': int }
        episode_vehicle_tracker = defaultdict(lambda: {'emergency_brakes': 0})
        
        for t in range(1, MAX_EPISODE_STEPS + 1):
            
            # 检测新生成的车辆并记录其开始时间
            current_vehicles = env._get_active_agents()
            for vid in current_vehicles:
                if vid not in episode_vehicle_tracker:
                    episode_vehicle_tracker[vid]['start_time'] = traci.simulation.getTime()

            if not states: break
                
            actions = {}
            for agent_id, state in states.items():
                # [关键] 在评估时，我们使用确定性动作，即取动作分布的均值
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    action_mean, _ = ppo_agent.policy.forward(state_tensor)
                    actions[agent_id] = action_mean.cpu().numpy().flatten()

            next_states, _, dones, _ = env.step(actions)
            
            # --- 实时数据收集 ---
            # 记录紧急刹车
            for agent_id in actions.keys():
                if agent_id in traci.vehicle.getIDList():
                    accel = traci.vehicle.getAcceleration(agent_id)
                    if accel < EMERGENCY_BRAKE_THRESHOLD:
                        episode_vehicle_tracker[agent_id]['emergency_brakes'] += 1

            # 处理已离开或碰撞的车辆
            colliding_ids = set(traci.simulation.getCollidingVehiclesIDList())
            all_current_ids = set(traci.vehicle.getIDList())
            
            for vid in list(episode_vehicle_tracker.keys()):
                # 检查是否已离开、碰撞或回合结束时仍在场
                is_done = vid not in all_current_ids
                
                if is_done or dones['__all__']:
                    stats = episode_vehicle_tracker.pop(vid)
                    stats['id'] = vid
                    stats['end_time'] = traci.simulation.getTime()
                    stats['travel_time'] = stats['end_time'] - stats['start_time']

                    if vid in colliding_ids:
                        stats['status'] = 'Collision'
                    elif is_done and vid not in colliding_ids: # 成功离开
                        stats['status'] = 'Success'
                    else: # 回合结束仍未离开
                        stats['status'] = 'Incomplete'
                    
                    all_vehicle_stats.append(stats)

            states = next_states
            if dones['__all__']: break
        
        total_simulation_time += traci.simulation.getTime()
        print(f"  > 回合 {i_episode}/{NUM_EVAL_EPISODES} 完成。")

    env.close()

    # --- 3. 计算评估仪表盘指标 ---
    print("\n--- 评估完成，生成仪表盘 ---")
    
    num_total_vehicles = len(all_vehicle_stats)
    if num_total_vehicles == 0:
        print("评估期间未出现车辆。")
        return

    successful_vehicles = [v for v in all_vehicle_stats if v['status'] == 'Success']
    collided_vehicles = [v for v in all_vehicle_stats if v['status'] == 'Collision']
    
    # -- 系统效率 (System Efficiency) --
    success_rate = (len(successful_vehicles) / num_total_vehicles) * 100 if num_total_vehicles > 0 else 0
    avg_travel_time = np.mean([v['travel_time'] for v in successful_vehicles]) if successful_vehicles else 0
    throughput_per_hour = (len(successful_vehicles) / total_simulation_time) * 3600 if total_simulation_time > 0 else 0
    
    # -- 系统安全性 (System Safety) --
    total_collisions = len(collided_vehicles)
    total_emergency_brakes = sum(v['emergency_brakes'] for v in all_vehicle_stats)
    
    # -- 系统公平性 (System Fairness) --
    successful_travel_times = [v['travel_time'] for v in successful_vehicles]
    fairness_index = jains_fairness_index(successful_travel_times)

    # --- 4. 打印仪表盘 ---
    print("\n==================== 评估仪表盘 ====================")
    print(f"模型: {model_path}")
    print(f"总评估回合数: {NUM_EVAL_EPISODES}")
    print(f"总车辆数: {num_total_vehicles}")
    print("\n--- 1. 系统效率 (System Efficiency) ---")
    print(f"  平均通行成功率 (Success Rate): {success_rate:.2f} %")
    print(f"  平均通行时间 (Avg Travel Time): {avg_travel_time:.2f} s")
    print(f"  系统吞吐量 (Throughput): {throughput_per_hour:.2f} 辆/小时")
    
    print("\n--- 2. 系统安全性 (System Safety) ---")
    print(f"  总碰撞次数 (Total Collisions): {total_collisions}")
    print(f"  总紧急刹车次数 (Total Emergency Brakes): {total_emergency_brakes}")
    
    print("\n--- 3. 系统公平性 (System Fairness) ---")
    print(f"  通行时间Jain公平指数 (Jain's Fairness Index): {fairness_index:.4f}")
    print("      (指数越接近1，代表通行时间越公平)")
    print("====================================================")


if __name__ == '__main__':
    evaluate(model_path,SUMO_CFG_FILE)