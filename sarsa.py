import gym
import numpy as np
import pickle
from pre_marsh_env import PreMarshEnv
from gym import spaces

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Box(0,env.total_slabs, shape=(env.observation_size,))
    def reset(self):
        state = self.env.reset()
        state = env.stateDictToArray(state)
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = env.stateDictToArray(obs)
        return obs, reward, done, info

env = CustomWrapper(PreMarshEnv())

def treina(passos):
    # Definir parâmetros de aprendizado
    alpha = 0.1 # Taxa de aprendizado
    gamma = 0.9 # Fator de desconto
    epsilon = 0.5 # Taxa de exploração

    # Inicializar a tabela de valores de ação (Q-Table)
    action_value_table = np.zeros((env.observation_space.shape[0], env.action_space.n))

    # Função para escolher uma ação baseada na política e-greedy
    def choose_action(state):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explorar
        else:
            i, j = np.unravel_index(action_value_table[state, :].argmax(), action_value_table.shape)
            #print (i,j)
            action = j #np.argmax(action_value_table[state, :]) # Explorar
        return action

    # Loop de episódios
    for episode in range(passos):
        state = env.reset()
        action = choose_action(state)
        done = False
        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = choose_action(next_state)
            # Atualizar a tabela de valores de ação (Q-Table)
            td_target = reward + gamma * action_value_table[next_state, next_action]
            td_error = td_target - action_value_table[state, action]
            action_value_table[state, action] += alpha * td_error
            # Atualizar estado e ação
            state = next_state
            action = next_action
        #print(episode)
    print(episode)
        
    # Salvar modelo gerado em um arquivo
    with open('modelo_sarsa_yard_89.pkl', 'wb') as f:
        pickle.dump(action_value_table, f)

def carrega():
    with open('modelo_sarsa_yard_89.pkl', 'rb') as f:
        action_value_table = pickle.load(f)

    # Fazer uma previsão para um determinado estado
    
    obs = env.reset()
    i = 0
    while i < 20:
        #action_values = action_value_table[obs]
        ix, jx = np.unravel_index(action_value_table[obs, :].argmax(), action_value_table.shape)
        action = jx
        #action = np.argmax(action_values)
        obs, rewards, dones, info = env.step(int(action))
        env.render(mode='console')
        print("Recompensa:", rewards)
        print("Objetivo:", env.objective)
        print("curr-move", env.current_action)
        print("Done:", dones)
        if dones:
            obs = env.reset()
            print(env.dones)

        i +=1

treina(1000000)
carrega()