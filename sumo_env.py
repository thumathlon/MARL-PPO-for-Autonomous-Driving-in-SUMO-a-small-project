# sumo_env.py (REFACTORED with Hybrid Reward and Low-Speed Penalty)
import os, sys, subprocess, gym
from gym import spaces
import numpy as np

# ... (SUMO_HOME check) ...
import traci
from traci.exceptions import TraCIException

## [MODIFIED] 奖励机制配置 ##
REWARD_CONFIG = {
    # 混合奖励权重 (Hybrid Reward Weights)
    "hybrid_weights": {
        # 提高个体权重，让智能体先学会开好自己的车
        "individual": 0.7, # w_ind: 之前是 0.3
        "social": 0.3      # w_soc: 之前是 0.7
    },
    
    # 个体奖励的内部权重 (Individual Reward Component Weights)
    "individual_weights": {
        "free_flow": 1.0,
        "comfort_accel": 0.5,
        "comfort_jerk": 0.5,
        "too_slow_penalty": 1.5 # [NEW] 为低速惩罚设置权重
    },
    
    # 社交奖励的内部权重 (Social Reward Component Weights)
    "social_weights": {
        "avg_speed": 1.0,
        # 大幅降低碰撞惩罚，避免智能体过度保守
        "collision_penalty": -100.0, # [MODIFIED] 之前是 -2000.0
    },

    # 目标值与阈值
    "targets": {
        "speed_limit": 33.33,
    },
    "thresholds": {
        "max_accel": 3.0,
        "max_jerk": 3.0,
        "speed_lower_bound": 10.0 # [NEW] 速度下限 (m/s)，约36km/h
    }
}

class SumoMultiAgentEnv(gym.Env):
    # ... (__init__, _start_simulation, reset, close, _get_active_agents, _get_state_for_agent 保持不变) ...
    def __init__(self, sumocfg_file, vehicle_prefixes=['main_', 'ramp_'], use_gui=False, max_episode_steps=3000):
        self.use_gui = use_gui
        self.max_episode_steps = max_episode_steps
        self.vehicle_prefixes = vehicle_prefixes
        
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.absolute_sumocfg_path = os.path.join(project_root, sumocfg_file)

        if not os.path.exists(self.absolute_sumocfg_path):
            raise FileNotFoundError(f"SUMO config file not found at: {self.absolute_sumocfg_path}")
            
        self.reward_config = REWARD_CONFIG
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        self.traci_connected = False
        
        self.last_speeds = {}
        self.last_accelerations = {}
        
    def _start_simulation(self):
        sumo_cmd = [self.sumo_binary, "-c", self.absolute_sumocfg_path]
        traci.start(sumo_cmd)
        self.traci_connected = True

    def reset(self):
        if self.traci_connected:
            self.close()
        self._start_simulation()
        self.steps = 0
        self.last_speeds.clear()
        self.last_accelerations.clear()
        
        for _ in range(5): traci.simulationStep()
        
        active_agents = self._get_active_agents()
        for agent_id in active_agents:
            if agent_id in traci.vehicle.getIDList():
                 speed = traci.vehicle.getSpeed(agent_id)
                 self.last_speeds[agent_id] = speed
                 self.last_accelerations[agent_id] = 0.0

        return {agent_id: self._get_state_for_agent(agent_id) for agent_id in active_agents if agent_id in traci.vehicle.getIDList()}
    
    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False

    def _get_active_agents(self):
        all_vehicles = traci.vehicle.getIDList()
        return [vid for vid in all_vehicles if any(vid.startswith(p) for p in self.vehicle_prefixes)]

    def _get_state_for_agent(self, vehicle_id):
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        if vehicle_id not in traci.vehicle.getIDList():
            return state

        ego_speed = traci.vehicle.getSpeed(vehicle_id)
        state[0] = ego_speed

        leader = traci.vehicle.getLeader(vehicle_id, 100.0)
        if leader and leader[0] != '':
            state[1] = leader[1]
            state[2] = traci.vehicle.getSpeed(leader[0]) - ego_speed
        else:
            state[1] = 100.0
            state[2] = 0.0

        follower = traci.vehicle.getFollower(vehicle_id, 100.0)
        if follower and follower[0] != '':
            state[3] = follower[1]
            state[4] = ego_speed - traci.vehicle.getSpeed(follower[0])
        else:
            state[3] = 100.0
            state[4] = 0.0
            
        return state
        
    def _calculate_social_reward(self, all_agent_ids):
        cfg = self.reward_config
        
        speeds = [traci.vehicle.getSpeed(agent_id) for agent_id in all_agent_ids if agent_id in traci.vehicle.getIDList()]
        avg_speed = np.mean(speeds) if speeds else 0
        r_avg_speed = avg_speed / cfg['targets']['speed_limit']

        num_collisions = traci.simulation.getCollidingVehiclesNumber()
        r_collision = num_collisions * cfg['social_weights']['collision_penalty']
        
        social_reward = (cfg['social_weights']['avg_speed'] * r_avg_speed + r_collision)
        
        return social_reward, {'social_avg_speed': r_avg_speed, 'social_collision_penalty': r_collision}

    ## [MODIFIED] 计算每个智能体的个体奖励 ##
    def _calculate_individual_reward(self, vehicle_id):
        if vehicle_id not in traci.vehicle.getIDList():
            return 0.0, {}
            
        cfg = self.reward_config
        weights = cfg['individual_weights']
        
        current_speed = traci.vehicle.getSpeed(vehicle_id)
        last_speed = self.last_speeds.get(vehicle_id, current_speed)
        acceleration = current_speed - last_speed
        
        last_accel = self.last_accelerations.get(vehicle_id, acceleration)
        jerk = acceleration - last_accel
        
        r_comfort_accel = 1 - min(abs(acceleration), cfg['thresholds']['max_accel']) / cfg['thresholds']['max_accel']
        r_comfort_jerk = 1 - min(abs(jerk), cfg['thresholds']['max_jerk']) / cfg['thresholds']['max_jerk']

        speed_error_ratio = abs(current_speed - cfg['targets']['speed_limit']) / cfg['targets']['speed_limit']
        r_free_flow = np.exp(-speed_error_ratio**2)
        
        # [NEW] 增加速度过低惩罚
        r_too_slow_penalty = 0
        if current_speed < cfg['thresholds']['speed_lower_bound']:
            # 使用二次方惩罚，速度越低，惩罚越大
            r_too_slow_penalty = (1.0 - current_speed / cfg['thresholds']['speed_lower_bound']) ** 2
        
        self.last_speeds[vehicle_id] = current_speed
        self.last_accelerations[vehicle_id] = acceleration
        
        individual_reward = (weights['free_flow'] * r_free_flow +
                             weights['comfort_accel'] * r_comfort_accel +
                             weights['comfort_jerk'] * r_comfort_jerk - # 注意是减去惩罚
                             weights['too_slow_penalty'] * r_too_slow_penalty)
                             
        sub_rewards = {
            'ind_free_flow': r_free_flow,
            'ind_comfort_accel': r_comfort_accel,
            'ind_comfort_jerk': r_comfort_jerk,
            'ind_too_slow_penalty': -weights['too_slow_penalty'] * r_too_slow_penalty # 记录惩罚值
        }

        return individual_reward, sub_rewards

    def step(self, actions):
        try:
            if not actions or not self._get_active_agents():
                traci.simulationStep()
                self.steps += 1
                return {}, {}, {'__all__': self.steps >= self.max_episode_steps}, {}

            for agent_id, action_value in actions.items():
                if agent_id in traci.vehicle.getIDList():
                    target_speed = (action_value[0] + 1) / 2 * self.reward_config['targets']['speed_limit']
                    traci.vehicle.setSpeed(agent_id, target_speed)
            
            traci.simulationStep()
            self.steps += 1
            
            active_agents = self._get_active_agents()
            observations, rewards, dones, infos = {}, {}, {}, {}

            social_reward, social_info = self._calculate_social_reward(active_agents)

            for agent_id in active_agents:
                observations[agent_id] = self._get_state_for_agent(agent_id)
                
                individual_reward, individual_info = self._calculate_individual_reward(agent_id)
                
                w_ind = self.reward_config['hybrid_weights']['individual']
                w_soc = self.reward_config['hybrid_weights']['social']
                
                final_reward = w_ind * individual_reward + w_soc * social_reward
                rewards[agent_id] = final_reward
                
                infos[agent_id] = {
                    'final_reward': final_reward,
                    'individual_reward': individual_reward,
                    'social_reward': social_reward,
                    **individual_info,
                    **social_info
                }

            dones = {agent_id: agent_id not in active_agents for agent_id in actions.keys()}
            dones['__all__'] = self.steps >= self.max_episode_steps or not active_agents

            return observations, rewards, dones, infos
        
        except TraCIException as e:
            print(f"Warning: TraCI connection lost or error occurred: {e}. Ending episode.")
            dones = {agent_id: True for agent_id in actions.keys()}
            dones['__all__'] = True
            self.close()
            return {}, {}, dones, {}