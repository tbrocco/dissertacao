import gymnasium
import os
import pygame
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
from pre_marsh_env import PreMarshEnv
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import QRDQN


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
        self.observation_space = spaces.Box(-np.inf,np.inf, shape=(env.observation_size,))
    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        state = self.env.stateDictToArray(state)
        return state, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = self.env.stateDictToArray(observation)
        #observation, reward, terminated, truncated, info = env.step(action)
        return observation, reward, terminated, truncated, info

def make_env(seed: int = 0):
    def _init():
        env_i = PreMarshEnv(num_stacks=8, stack_height=8, discreeteAction=True, max_episode_steps=1000, objective_size=7, render_mode='console')
        env = CustomWrapper(env_i)
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init

def create_custom_env(params):
    return CustomWrapper(PreMarshEnv(**params))

# Função para criar uma instância única do ambiente customizado com parâmetros
# def make_env(num_stacks, stack_height, discreeteAction, max_episode_steps, objective_size, render_mode):
#     return CustomWrapper(PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=discreeteAction, max_episode_steps=max_episode_steps, objective_size=objective_size, render_mode=render_mode))

def treina(passos, tipo, env_p, file_path="C:/temporario/modeloIA/yard_model_"):
    #env = PreMarshEnv()
    # env.reset()
    #env = CustomWrapper(env_p)


    #nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    nome_modelo = file_path + tipo
    if tipo == "dqn":
        # Define o modelo DQN
        model = DQN('MlpPolicy',env_p, 
                    #learning_rate=1e-3
                    buffer_size=524288, batch_size=8192, 
                    learning_starts=1, train_freq=4, target_update_interval=8,
                    exploration_fraction=0.7, exploration_final_eps=0.2, verbose=1,
                    device='cuda', tensorboard_log="./yard_tensorboard/"
                    )
        # Cria um callback para salvar o modelo a cada 10000 steps
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn' )

        model.learn(total_timesteps=passos , log_interval=50, progress_bar=True, tb_log_name="dqn")
    elif tipo == "QRDQN":
        policy_kwargs = dict(n_quantiles=50)
        model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1 ,device='cuda', tensorboard_log="./yard_tensorboard/", target_update_interval=8, batch_size=256, buffer_size=131072)
        model.learn(total_timesteps=passos, log_interval=1, progress_bar=True, tb_log_name="QRDQN")
    elif tipo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1
                    , device="cuda", batch_size=64
                    , tensorboard_log="./yard_tensorboard/")
        model.learn(total_timesteps=passos, progress_bar=True, tb_log_name="ppo")
    else:
        return print("tipo invalido")
    
    model.save(nome_modelo)

def carrega(tipo, env_p, avalia=False, file_path="C:/temporario/modeloIA/yard_model_"):
    #nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    nome_modelo = file_path + tipo
    if tipo == "dqn":
        # Carrega o modelo treinado
        model = DQN.load(nome_modelo)
    elif tipo == "QRDQN":
        # Carrega o modelo treinado
        model = QRDQN.load(nome_modelo)
    elif tipo == "PPO":
        model = PPO.load(nome_modelo)
    else:
        return print("tipo invalido")

    # Cria o ambiente
    env = CustomWrapper(env_p) #PreMarshEnv(render_mode='console'))
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

def render_episode(env, tipo,file_path="C:/temporario/modeloIA/yard_model_"):

    #nome_modelo = "C:/temporario/modeloIA/yard_model_" + tipo
    nome_modelo = file_path + tipo
    if tipo == "dqn":
        # Carrega o modelo treinado
        model = DQN.load(nome_modelo)
    if tipo == "QRDQN":
        # Carrega o modelo treinado
        model = QRDQN.load(nome_modelo)
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
        action, _ = model.predict(obs, deterministic=False)

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


def renderizaImagens(tipo, env_p,file_path):
    # Crie o ambiente
    env = CustomWrapper(env_p) #PreMarshEnv(render_mode='console'))
    

    # Defina aqui o diretório onde as imagens serão salvas
    output_directory = 'output_images'

    # Crie o diretório de saída se ele não existir
    os.makedirs(output_directory, exist_ok=True)

    for ep_i in range(10):
        # Renderiza o episódio e obtém a imagem concatenada
        episode_image = render_episode(env, tipo,file_path)
        #episode_image= np.rot90(episode_image, k=1)
        seed = ep_i
        # Salva a imagem em um arquivo
        img_filename = os.path.join(output_directory, f"episode_image_{tipo}{ep_i}.png")
        pygame.image.save(pygame.surfarray.make_surface(episode_image), img_filename)

    # Fecha o ambiente
    env.close()

tipo = "dqn" #QRDQN #dqn #PPO                                                                  pooo0 
#default_occupancy=0.5
num_stacks = 5
stack_height = 5
objective_size= 4
optimal_solution = ((int(stack_height/1)+stack_height) * objective_size)
max_episode_steps = optimal_solution + 1
max_episode_steps = 100

# Parâmetros a serem passados para make_env
env_params_list = [
    {
    'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    }
    # {
    # 'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    # }
    # {
    # 'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    # },
    # {
    # 'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    # }
]



file_path=f"C:/temporario/modeloIA/yard_model_capacity_{objective_size}objetivos_tamanhoDopatio_{num_stacks}x{stack_height}"

env = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=True, max_episode_steps=max_episode_steps, objective_size=objective_size, render_mode='console')
# Crie o ambiente vetorizado usando make_vec_env
#env_vec = make_vec_env(make_env, n_envs=len(env_params_list), env_kwargs=env_params_list)
# Crie o ambiente vetorizado usando make_vec_env
#env_vec = ([lambda: make_env(env_params) for env_params in env_params_list])


# Create the vectorized environment
#vec_env = SubprocVecEnv([make_env(i) for i in range(8)])
vec_env = DummyVecEnv([lambda: Monitor(create_custom_env(env_params), "./yard_tensorboard") for env_params in env_params_list])

treina(3000000, tipo, env_p=vec_env, file_path=file_path)
#treina(1000000, "PPO", env_p=env, file_path=file_path)

env = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=True, max_episode_steps=max_episode_steps, objective_size=objective_size, render_mode='console')
#carrega("PPO", avalia=True, env_p=env)
carrega(tipo=tipo, avalia=True, env_p=env, file_path=file_path)
env = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=True, max_episode_steps=max_episode_steps,  objective_size=objective_size, render_mode='rgb_array')
renderizaImagens(tipo, env_p=env,file_path=file_path)
#renderizaImagens("PPO", env_p=env)
