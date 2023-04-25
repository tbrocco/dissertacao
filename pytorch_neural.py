import torch
import torch.nn as nn
import torch.optim as optim
import gym
import pre_marsh_env
import torch.nn.functional as F

# Define a rede neural
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


# Defina os hiperparâmetros
learning_rate = 0.001
gamma = 0.99
hidden_size = 128

# Cria o ambiente
env = pre_marsh_env.PreMarshEnv()

# Obtém o tamanho da entrada e saída da rede neural
input_size = env.observation_size
output_size = env.action_space.n

# Instancia a rede neural e define o otimizador
policy_network = PolicyNetwork(input_size, hidden_size, output_size)
optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)

# Função de seleção de ação
# def select_action(state):
#     state = torch.from_numpy(state).float().unsqueeze(0)
#     probs = policy_network(state)
#     c = torch.distributions.Categorical(probs=probs)
#     action = c.sample()
#     return action.item()

def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0)#.to(device="cuda")
    probs = policy_network(state)
    probs = torch.clamp(probs, min=-1e6, max=1e6) # limita a saída do modelo entre -1e6 e 1e6
    probs = F.softmax(probs, dim=-1)
    c = torch.distributions.Categorical(probs=probs)
    action = c.sample()
    return action.item()

# Treinamento
def train(episodes):
    for episode in range(episodes):
        state = env.reset()
        state = env.stateDictToArray(state)
        rewards = []
        actions = []
        states = []

        # Executa o episódio
        while True:
            action = select_action(state)
            new_state, reward, done, _ = env.step(action)
            new_state = env.stateDictToArray(new_state)

            rewards.append(reward)
            actions.append(action)
            states.append(state)

            if done:
                # Calcula a recompensa acumulada ao longo do episódio
                R = 0
                for r in rewards[::-1]:
                    R = r + gamma * R

                # Calcula as perdas
                loss = 0
                for s, a, r in zip(states, actions, rewards):
                    state_tensor = torch.from_numpy(s).float().unsqueeze(0)
                    action_tensor = torch.tensor(a).unsqueeze(0)
                    #transformar em 1d
                    action_tensor = action_tensor.view(-1, 1)
                    #fim
                    R_tensor = torch.tensor(R).unsqueeze(0)

                    probs = policy_network(state_tensor)
                    log_probs = torch.log(probs)
                    selected_log_prob = log_probs.gather(1, action_tensor)
                    loss += -selected_log_prob * (R_tensor - 0)

                # Realiza o backpropagation e atualiza os pesos da rede neural
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

            state = new_state

def test_model(env, model):
    state = env.reset()
    state = env.stateDictToArray(state)
    done = False
    total_reward = 0

    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = model(state)
        action = torch.argmax(action_probs, dim=1)
        next_state, reward, done, _ = env.step(action.item())
        next_state = env.stateDictToArray(next_state)
        state = next_state
        total_reward += reward
        # if done :
        #     state = env.reset()
        #     state = env.stateDictToArray(state)
        #     total_reward = 0

    return total_reward

def test(env, model, num_episodes=10):
    total_rewards = []
    for i in range(num_episodes):
        reward = test_model(env, model)
        total_rewards.append(reward)
    avg_reward = sum(total_rewards) / num_episodes
    print("Average reward over {} test episodes: {}".format(num_episodes, avg_reward))

def verifica(env, model, num_episodes=10):
    state = env.reset()
    state = env.stateDictToArray(state)
    done = False
    total_reward = 0

    for i in range(num_episodes):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = model(state)
        action = torch.argmax(action_probs, dim=1)
        next_state, reward, done, _ = env.step(action.item())
        next_state = env.stateDictToArray(next_state)
        state = next_state
        if done :
             state = env.reset()
             state = env.stateDictToArray(state)
        print(reward, done)
        
train(500000)
torch.save(policy_network.state_dict(), 'policy.pth')
policy_network.load_state_dict(torch.load('policy.pth'))
#test(env, policy_network, 10)

verifica(env, policy_network, 10)
