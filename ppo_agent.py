# ppo_agent.py (MODIFIED)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal

# ############################## 经验内存 (无需修改) ##############################
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.batch_size = batch_size

    def store(self, state, action, log_prob, reward, done, val):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.vals.append(val)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return torch.tensor(np.array(self.states), dtype=torch.float32), \
               torch.tensor(np.array(self.actions), dtype=torch.float32), \
               torch.tensor(np.array(self.log_probs), dtype=torch.float32), \
               torch.tensor(np.array(self.rewards), dtype=torch.float32), \
               torch.tensor(np.array(self.dones), dtype=torch.float32), \
               torch.tensor(np.array(self.vals), dtype=torch.float32), \
               batches

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.vals = []

# ############################## Actor-Critic 网络 (无需修改) ##############################
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * action_std_init)

    def get_action_std(self):
        return torch.exp(self.action_log_std)

    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value

    def act(self, state):
        action_mean, _ = self.forward(state)
        action_std = self.get_action_std()
        
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(state)
        
        action = torch.clamp(action, -1, 1)
        
        return action.detach().numpy().flatten(), log_prob.detach(), value.detach()

    def evaluate(self, state, action):
        action_mean, value = self.forward(state)
        action_std = self.get_action_std()
        dist = Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        return log_prob, value.squeeze(-1), entropy


# ############################## PPO 智能体 (已修改) ##############################
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, batch_size, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # [MODIFIED] 使用可变的action_std
        self.action_std = action_std_init

        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                        # [MODIFIED] 不再将action_log_std作为可训练参数
                        # {'params': self.policy.action_log_std, 'lr': lr_actor}
                    ])
        
        self.memory = PPOMemory(batch_size)
        self.MseLoss = nn.MSELoss()

    # [NEW] 增加动作标准差衰减的函数
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.action_log_std.data.fill_(np.log(self.action_std))
        self.policy_old.action_log_std.data.fill_(np.log(self.action_std))

    def decay_action_std(self, decay_rate, min_std):
        self.action_std = max(self.action_std - decay_rate, min_std)
        self.set_action_std(self.action_std)
        return self.action_std

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action, log_prob, value = self.policy_old.act(state_tensor)
        
        return action, log_prob, value

    def update(self):
        # ... (update函数内部逻辑保持不变，它将自动使用新的action_std) ...
        states, actions, old_log_probs, rewards, dones, vals, batches = self.memory.generate_batches()
        
        advantages = torch.zeros_like(rewards)
        gae = 0
        with torch.no_grad():
            last_val = self.policy_old.critic(states[-1]).squeeze()

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_val = vals[t+1].squeeze() if t < len(rewards) - 1 else last_val
            
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - vals[t].squeeze()
            gae = delta + self.gamma * 0.95 * gae * next_non_terminal
            advantages[t] = gae
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                batch_vals = vals[batch_indices].squeeze(-1)
                batch_returns = batch_advantages + batch_vals

                new_log_probs, new_values, entropy = self.policy.evaluate(batch_states, batch_actions)

                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()

                v_loss_unclipped = self.MseLoss(new_values, batch_returns)
                v_clipped = batch_vals + torch.clamp(new_values - batch_vals, -self.eps_clip, self.eps_clip)
                v_loss_clipped = self.MseLoss(v_clipped, batch_returns)
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped)

                loss = actor_loss + critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()
        
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(torch.load(path))