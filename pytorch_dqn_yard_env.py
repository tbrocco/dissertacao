import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from yard_env import YardEnv

# Define the Deep Q-Network (DQN) class
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        #x = x.view(x.size(0), -1)  # adiciona uma camada de achatamento
        x = torch.relu(self.fc1(x)) # aplica a transformação linear seguida de ReLU
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size, lr, gamma, epsilon, epsilon_decay):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=10000)
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def act(self, state):
        if np.random.random() < self.epsilon:
            #return np.random.choice(self.action_size)
            origem = np.random.choice(self.action_size)
            destino = np.random.choice(self.action_size)
            quantidade = np.random.choice(3)
        else:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
                #return np.argmax(q_values.numpy())
                origem = np.argmax(q_values.numpy())
                quantidade = np.random.choice(3)
                destino = np.argmax(q_values[0][origem*quantidade:(origem+1)*quantidade].numpy())
        return (origem, destino, quantidade )
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size:
    #         return
    #     batch = np.random.choice(len(self.memory), batch_size, replace=False)
    #     states = np.array([transition[0] for transition in batch])
    #     actions = np.array([transition[1] for transition in batch])
    #     rewards = np.array([transition[2] for transition in batch])
    #     next_states = np.array([transition[3] for transition in batch])
    #     dones = np.array([transition[4] for transition in batch])
    #     states = torch.from_numpy(states).float()
    #     actions = torch.from_numpy(actions).long()
    #     rewards = torch.from_numpy(rewards).float()
    #     next_states = torch.from_numpy(next_states).float()
    #     dones = torch.from_numpy(dones).float()
    #     q_values = self.model(states)
    #     next_q_values = self.model(next_states)
    #     q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    #     next_q_values = next_q_values.max(1)[0]
    #     targets = rewards + (1 - dones) * self.gamma * next_q_values
    #     loss = self.criterion(q_values, targets.detach())
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        transitions = [self.memory[i] for i in batch] # obtém as transições do índice aleatório
        states = np.array([np.array(transition[0]) for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([np.array(transition[3]) for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).float()
        #next_states = next_states.astype(np.float32)
        #next_states = torch.from_numpy(next_states)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float()
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

# Define the main function to train and test the DQN Agent
if __name__ == '__main__':
    env = YardEnv()
    state_size = env.num_stacks * env.stack_height #env.observation_space.shape[0]
    action_size = env.num_stacks-1 #(env.num_stacks, env.stack_height, 3) #env.action_space.n
    agent = DQNAgent(state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999)
    episodes = 1000
    batch_size = 32
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            agent.replay(batch_size)
            agent.decay_epsilon()
        print(f"Episode {episode} - Score: {score} - Epsilon: {agent.epsilon:.2f}")
    agent.save("model.pt")
    env.close()
