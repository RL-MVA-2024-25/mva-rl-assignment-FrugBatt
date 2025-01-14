import torch
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from random import random, randint, sample
import numpy as np
from model import DQNModel

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class DQNAgent:

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, observation, use_random=False, grad=False, return_pt=True):
        if use_random:
            return randint(0, 3)
            
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)

        if grad:
            q_values = self.model(observation).argmax()
        else:
            with torch.no_grad():
                q_values = self.model(observation).argmax().item()

        if return_pt:
            return q_values
        return q_values.item()

    def save(self, path="models/dqn.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"DQN Model saved at {path}")

    def load(self):
        model_path = "models/dqn.pth"
        self.model.load_state_dict(torch.load(model_path))
        print(f"DQN Model loaded from {model_path}")


class Memory:

    def __init__(self, max_size, device=None):
        self.max_size = max_size
        self.size = 0
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.state_memory = torch.empty((max_size, 6)).to(self.device)
        self.action_memory = torch.empty((max_size, 1), dtype=torch.long).to(self.device)
        self.reward_memory = torch.empty((max_size, 1)).to(self.device)
        self.next_state_memory = torch.empty((max_size, 6)).to(self.device)

    def push(self, state, action, reward, next_state):
        index = self.size % self.max_size
        self.state_memory[index] = torch.tensor(state).to(self.device)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = torch.tensor(next_state).to(self.device)
        self.size += 1

    def sample(self, batch_size):
        indices = torch.randint(0, self.size if self.size < self.max_size else self.max_size, (batch_size,))
        return self.state_memory[indices], self.action_memory[indices], self.reward_memory[indices], self.next_state_memory[indices]

    def clear(self):
        self.size = 0

    def __len__(self):
        return self.size if self.size < self.max_size else self.max_size

class DQNTrainer:

    def __init__(self, agent, target_model, env, device=None, gamma=0.99, lr=1e-3, batch_size=64, memory_size=100000, epsilon_start=1.0, epsilon_delay=500, epsilon_end=10000):
        self.agent = agent
        self.target_model = target_model
        self.env = env
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = Memory(memory_size)
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()
        self.epsilon = epsilon_start
        self.epsilon_delay = epsilon_delay
        self.epsilon_end = epsilon_end
        self.best_reward = 0

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state = self.memory.sample(self.batch_size)

        # state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        # action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)
        # reward = torch.tensor(np.array(reward), dtype=torch.float32).to(self.device)
        # next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)

        q_values = self.agent.model(state)
        next_q_values = self.target_model(next_state).detach()

        q_value = q_values.gather(1, action).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward.squeeze() + self.gamma * next_q_value

        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_epoch(self, use_random=False):
        state, _ = self.env.reset()
        total_reward = 0
        losses = []

        for _ in range(200):
            action = self.agent.act(state, use_random)
            next_state, reward, _, _, _ = self.env.step(action)
            total_reward += reward

            self.memory.push(state, action, reward, next_state)

            loss = self.optimize_model()
            if loss is not None:
                losses.append(loss)

            state = next_state

        return total_reward, losses
    
    def train(self, n_ep=1000):
        self.agent.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.agent.model.state_dict())

        for ep in range(n_ep):
            if ep < self.epsilon_end:
                if ep % self.epsilon_delay == 0:
                    self.epsilon = max(0.05, self.epsilon - 0.05)
            elif ep == self.epsilon_end:
                self.epsilon = 0
            total_reward, losses = self.run_epoch(use_random=((self.epsilon > 0) and (random() < self.epsilon)))

            if total_reward >= 5e10:
                grade = 6
            elif total_reward >= 2e10:
                grade = 5
            elif total_reward >= 1e10:
                grade = 4
            elif total_reward >= 1e9:
                grade = 3
            elif total_reward >= 1e8:
                grade = 2
            elif total_reward >= 3432807.680391572:
                grade = 1
            else:
                grade = 0

            print(f"Episode {ep + 1} - Reward: {total_reward} - Loss: {sum(losses) / len(losses)} - Grade: {grade}/6")

            if (ep + 1) % 30 == 0:
                self.target_model.load_state_dict(self.agent.model.state_dict())

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.agent.save()
                print(f"Best reward so far: {total_reward}")

class ProjectAgent(DQNAgent):

    def __init__(self):
        model = DQNModel()
        super().__init__(model, device='cpu')
