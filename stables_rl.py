import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from gym import spaces
from pre_marsh_env import PreMarshEnv
import torch.nn as nn
from stable_baselines3 import PPO

# import matplotlib
# matplotlib.use('TkAgg',force=True)
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomPolicy, self).__init__()
        self.fc1 = nn.Linear(observation_space.shape[0], 64)
        self.fc2 = nn.Linear(64, action_space.n)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.velho = env
        print(env.observation_space)
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




def treina(passos, tipo):
    # env = PreMarshEnv(10,11,True)
    # env.reset()
    env = CustomWrapper(PreMarshEnv())

    # Cria um ambiente vetorizado com 4 ambientes paralelos
    env = make_vec_env(lambda:  CustomWrapper(PreMarshEnv()), n_envs=8)

    nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    
    if tipo == "dqn":
        # Define o modelo DQN
        model = DQN('MlpPolicy',env, 
                    learning_rate=1e-3, buffer_size=10000, batch_size=64, 
                    learning_starts=1000, train_freq=4, target_update_interval=1000, 
                    exploration_fraction=0.2, exploration_final_eps=0.3, verbose=1,
                    device='cuda')
        # Cria um callback para salvar o modelo a cada 10000 steps
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn')
        model.learn(total_timesteps=passos, log_interval=1000, progress_bar=True)
        
    elif tipo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device="cuda")
        model.learn(total_timesteps=passos, progress_bar=True)
    else:
        return print("tipo invalido")
    
    model.save(nome_modelo)

    # # Define o modelo DQN
    # model = DQN('MlpPolicy',env, 
    #             learning_rate=1e-3, buffer_size=10000, batch_size=64, 
    #             learning_starts=1000, train_freq=4, target_update_interval=1000, 
    #             exploration_fraction=0.1, exploration_final_eps=0.1, verbose=1, 
    #             device='cuda')

    # # Treina o modelo por 1000000 de steps
    # model.learn(total_timesteps=passos, callback=checkpoint_callback, progress_bar=True, )



    # Avalia o modelo
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
    del model # remove to demonstrate saving and loading    

def carrega(tipo, avalia=False):
    nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    if tipo == "dqn":
        # Carrega o modelo treinado
        model = DQN.load(nome_modelo)
    elif tipo == "PPO":
        model = PPO.load(nome_modelo)
    else:
        return print("tipo invalido")

    # Cria o ambiente
    env = CustomWrapper(PreMarshEnv())

    

    # Executa previsões com o modelo carregado
    obs = env.reset()
    #print(obs)
    env.render(mode='console')

    if (avalia==True) : 
        mean_reward, std_reward = evaluate_policy(model, env, render=False, n_eval_episodes=50)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        i = 0
        while i < 20:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            env.render(mode='console')
            print("Recompensa:", reward)
                #env.printa()
            if done:
                env.reset()
                print(env.dones)
                print("fim")
                print("---------------------------------------------")
            i +=1

    env.close()



#treina(5000000, "PPO")
treina(200000, "dqn")
#carrega("PPO")
carrega("dqn")

#--python -m trace --trace YOURSCRIPT.py