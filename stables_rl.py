import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
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
        self.fc1 = nn.Linear(observation_space.shape[0], 8)
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

def create_custom_env(params):
    return CustomWrapper(PreMarshEnv(**params))

env = CustomWrapper(PreMarshEnv())

env_params = [
    {'default_occupancy': 0.3},  # Parâmetros do ambiente 1
    {'default_occupancy': 0.4},  # Parâmetros do ambiente 2
    {'default_occupancy': 0.5},  # Parâmetros do ambiente 3
    {'default_occupancy': 0.6},  # Parâmetros do ambiente 4
    {'default_occupancy': 0.7},  # Parâmetros do ambiente 5
    {'default_occupancy': 0.75},  # Parâmetros do ambiente 6
    {'default_occupancy': 0.8},  # Parâmetros do ambiente 7
    {'default_occupancy': 0.85}  # Parâmetros do ambiente 8
    # Adicione mais parâmetros para cada ambiente, se necessário
]



def treina(passos, tipo):
    # env = PreMarshEnv()
    # env.reset()
    #env = CustomWrapper(PreMarshEnv())

    # # # Cria um ambiente vetorizado com 4 ambientes paralelos
    # envs = [lambda params=params: create_custom_env(params)  for params in env_params]
    # #env = make_vec_env(envs, n_envs=len(env_params))
    # env = make_vec_env(lambda: envs, n_envs=len(env_params))
    env = make_vec_env(lambda:  CustomWrapper(PreMarshEnv()), n_envs=8 )

    nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    
    if tipo == "dqn":
        # Define o modelo DQN
        model = DQN('MlpPolicy',env, 
                    learning_rate=1e-3, buffer_size=10000, batch_size=128, 
                    learning_starts=1000, train_freq=4, target_update_interval=1000, 
                    exploration_fraction=0.2, exploration_final_eps=0.3, verbose=1,
                    device='cuda')
        # Cria um callback para salvar o modelo a cada 10000 steps
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn' )

        model.learn(total_timesteps=passos, log_interval=1000, progress_bar=True)
        
    elif tipo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log="./ppo_yard_tensorboard/")
        model.learn(total_timesteps=passos, progress_bar=True, tb_log_name="first_run")
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
        while i < 3:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            env.render(mode='console')
            print("Recompensa:", reward)
                #env.printa()
            if done:
                i +=1
                env.reset()
                print(env.dones)
                print("fim")
                print("---------------------------------------------")
            
    env.close()



treina(4000000, "PPO")
#treina(100000, "dqn")
carrega("PPO")
#carrega("dqn")

#--python -m trace --trace YOURSCRIPT.py