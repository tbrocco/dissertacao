import gymnasium
import os
import pygame
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import spaces
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

class CustomWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.velho = env
        print(env.observation_space)
        self.observation_space = spaces.Box(0,env.total_slabs_max, shape=(env.observation_size,))
    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        state = self.env.stateDictToArray(state)
        return state, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.env.stateDictToArray(observation)
        #observation, reward, terminated, truncated, info = env.step(action)
        return observation, reward, terminated, truncated, info
    


def create_custom_env(params):
    return CustomWrapper(PreMarshEnv(**params))

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
    #env = PreMarshEnv()
    # env.reset()
    env = CustomWrapper(PreMarshEnv(8,8,True))


    # env = gymnasium.vector.SyncVectorEnv([
    #     lambda: CustomWrapper(PreMarshEnv(default_occupancy=0.3)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.4)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.5)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.6)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.65)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.7)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.75)),
    #     lambda: CustomWrapper(PreMarshEnv(8,8,True, 0.8)),
    # ])

    # # # Cria um ambiente vetorizado com 4 ambientes paralelos
    #envs = [lambda params=params: create_custom_env(params)  for params in env_params]
    # env = make_vec_env(envs, n_envs=len(env_params))
    #env = make_vec_env(lambda: envs, n_envs=len(env_params))
    #env = make_vec_env(lambda:  CustomWrapper(PreMarshEnv()), n_envs=8 )

    nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    
    if tipo == "dqn":
        # Define o modelo DQN
        model = DQN('MlpPolicy',env, 
                    learning_rate=1e-3, buffer_size=10000, batch_size=64, 
                    learning_starts=1000, train_freq=4, target_update_interval=1000, 
                    exploration_fraction=0.2, exploration_final_eps=0.3, verbose=1
                    ,device='cuda', tensorboard_log="./yard_tensorboard/"
                    )
        # Cria um callback para salvar o modelo a cada 10000 steps
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn' )

        model.learn(total_timesteps=passos, log_interval=1000, progress_bar=True, tb_log_name="dqn")
        
    elif tipo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1
                    , device="cuda", batch_size=64
                    , tensorboard_log="./yard_tensorboard/")
        model.learn(total_timesteps=passos, progress_bar=True, tb_log_name="ppo")
    else:
        return print("tipo invalido")
    
    model.save(nome_modelo)

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
    env = CustomWrapper(PreMarshEnv(render_mode='console'))

    # Executa previsões com o modelo carregado
    obs, info = env.reset()
    #print(obs)
    env.render()

    if (avalia==True) : 
        mean_reward, std_reward = evaluate_policy(model, env, render=False, n_eval_episodes=10)
        print(f"{tipo}: - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        i = 0
        while i < 3:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            env.render()
            done = truncated + terminated
            print("Recompensa:", reward)
                #env.printa()
            if done:
                i +=1
                env.reset()
                print(env.dones)
                print("fim")
                print("---------------------------------------------")
            
    env.close()

def render_episode(env, tipo):

    nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    if tipo == "dqn":
        # Carrega o modelo treinado
        model = DQN.load(nome_modelo)
    elif tipo == "PPO":
        model = PPO.load(nome_modelo)

    # Defina aqui as configurações do Pygame
    # Define RENDER
    square_size = 20
    margin = 5
    marginleft = 15
    toolbar = 40
    footer = 30

    num_stacks = env.TamanhoPatio
    stack_height = env.TamanhoPilha
    screen_width =  num_stacks * (square_size + margin) + margin + toolbar
    screen_width = screen_width*10
    screen_height = screen_height = (stack_height + 1) * (square_size + margin) + margin + toolbar + footer
    screen_height = screen_height*10

    obs, info = env.reset()
    frames = []
    # Renderiza o ambiente
    #mode='rgb_array'
    img = env.render()

    # Corrige a inversão da imagem
    img = np.fliplr(img)
    # Adiciona o quadro renderizado à lista de frames
    frames.append(img)

    done = False
    while not done:
      
        # Exe
        # ta uma ação no ambiente
        #action = env.action_space.sample()
        action, _ = model.predict(obs, deterministic=True)

        #obs, reward, done, info = env.step(action)
        obs, reward, terminated, truncated, info = env.step(int(action))
        # Renderiza o ambiente
        img = env.render()

        # Corrige a inversão da imagem
        img = np.fliplr(img)
        # Adiciona o quadro renderizado à lista de frames
        frames.append(img)
        done = terminated + truncated

    # Concatena os frames horizontalmente para criar a imagem do episódio
    episode_image = np.concatenate(frames, axis=1)
    episode_image= np.rot90(episode_image, k=1)

    return episode_image


def renderizaImagens(tipo):
    # Crie o ambiente
    env = CustomWrapper(PreMarshEnv(render_mode='rgb_array'))

    # Defina aqui o diretório onde as imagens serão salvas
    output_directory = 'output_images'

    # Crie o diretório de saída se ele não existir
    os.makedirs(output_directory, exist_ok=True)

    for ep_i in range(10):
        # Renderiza o episódio e obtém a imagem concatenada
        episode_image = render_episode(env, tipo)
        #episode_image= np.rot90(episode_image, k=1)
        seed = ep_i
        # Salva a imagem em um arquivo
        img_filename = os.path.join(output_directory, f"episode_image_{tipo}{ep_i}.png")
        pygame.image.save(pygame.surfarray.make_surface(episode_image), img_filename)

    # Fecha o ambiente
    env.close()

#treina(1000000, "dqn")
#treina(1000000, "PPO")
carrega("PPO", True)
carrega("dqn", True)
renderizaImagens("PPO")
renderizaImagens("dqn")

