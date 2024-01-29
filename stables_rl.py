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
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3 import PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from typing import Callable
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value:
    :return: current learning rate depending on remaining progress
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def get_space_size(space):
    if isinstance(space, gymnasium.spaces.Dict):
        return sum([get_space_size(s) for s in space.spaces.values()])
    elif isinstance(space.shape, tuple):
        return np.prod(space.shape)
    else:
        return space.shape
        
# import matplotlib
# matplotlib.use('TkAgg',force=True)
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())
class CustomPolicy(MultiInputActorCriticPolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):

        super(CustomPolicy, self).__init__(observation_space, action_space, *args, **kwargs)

        # Defina sua arquitetura personalizada aqui
        self.features_dim = get_space_size(self.observation_space) 

        # Definir sua arquitetura personalizada aqui
        self.pi_fc0 = nn.Linear(self.features_dim, 64)
        self.pi_fc1 = nn.Linear(64, 64)
        self.pi_fc2 = nn.Linear(64, self.action_space.n)

        self.vf_fc0 = nn.Linear(self.features_dim, 64)
        self.vf_fc1 = nn.Linear(64, 64)
        self.vf_fc2 = nn.Linear(64, 1)

        self.activation_fn = nn.ReLU()

    def forward(self, obs, action, seq_lens, use_sde=False):
    #def forward(self, obs, state, mask, use_sde=False):
        features = self.extract_features(obs)
        
        pi = self.activation_fn(self.pi_fc0(features))
        pi = self.activation_fn(self.pi_fc1(pi))
        pi = self.pi_fc2(pi)

        vf = self.activation_fn(self.vf_fc0(features))
        vf = self.activation_fn(self.vf_fc1(vf))
        vf = self.vf_fc2(vf)

        return pi, vf

class CustomActionWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # self.velho = env
        # print(env.observation_space)
        # self.observation_space = spaces.Box(-np.inf,np.inf, shape=(env.observation_size,))
    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        # state = self.env.stateDictToArray(state)
        return state, info
    def mask_action(self, action):
        current_action = self.env.ACTION_LOOKUP[action] #(x,y,w)
        src_stack, dst_stack, num_slabs = current_action
        num_slabs = num_slabs+1

        # Verifica se a origem tem placa suficiente
        valido2_origem, delta_origem = self.valida2_Origem(src_stack, num_slabs)

        # Verifica se o destino cabe as placas
        valido2_destino, delta_destino = self.valida2_Destino(dst_stack, num_slabs)

        # Verifica se a ação é válida usando a lógica de validação 3
        valido3 = self.valida3(src_stack, dst_stack, num_slabs)

        # Aplica a máscara de ação
        if not valido2_origem or not valido2_destino or not valido3:
            return None  # Retorna None para indicar que a ação é inválida
        else:
            # A ação é válida, retorna a ação original
            return action
        
    def step(self, action):
        # Aplica a máscara de ação às ações
        masked_action = self.mask_action(action)
        if masked_action is None:
            # Ação inválida, retorne o estado atual e uma recompensa inválida
            return self.state, -100, False, False, {}

        observation, reward, terminated, truncated, info = self.env.step(action)
        # observation = self.env.stateDictToArray(observation)
        # #observation, reward, terminated, truncated, info = env.step(action)
        return observation, reward, terminated, truncated, info

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
        #model = DQN('MultiInputPolicy',env_p, 
        model = DQN('MlpPolicy',env_p, 
                    #learning_rate=1e-3
                    #buffer_size=1048576, batch_size=32768,
                    buffer_size=4096, batch_size=512, 
                    learning_starts=1, train_freq=8, target_update_interval=32,
                    exploration_fraction=0.7, exploration_final_eps=0.2, verbose=1,
                    device='cuda', tensorboard_log="./yard_tensorboard/"
                    )
        # Cria um callback para salvar o modelo a cada 10000 steps
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn' )

        model.learn(total_timesteps=passos,  log_interval=50, progress_bar=True, tb_log_name="dqn")
    elif tipo == "QRDQN":
        policy_kwargs = dict(n_quantiles=50)
        model = QRDQN("MultiInputPolicy", env_p, policy_kwargs=policy_kwargs, verbose=1 ,device='cuda', tensorboard_log="./yard_tensorboard/", target_update_interval=8, batch_size=256, buffer_size=131072)
        model.learn(total_timesteps=passos, log_interval=1, progress_bar=True, tb_log_name="QRDQN")
    elif tipo == "PPO":
        #custom_policy = CustomPolicy(env_p.observation_space, env_p.action_space, lr_schedule=linear_schedule(0.0003))
        # Definição da arquitetura da rede usando net_arch
        # net_arch = [{'pi': [env_p.observation_size, 64], 'vf': [env_p.action_space.n, 64]}]
        # net_arch = [{'pi': [env_p.observation_size, {'pi_fc0': 64, 'pi_fc1': 64}, 64, env_p.action_space.n], 'vf': [env_p.observation_size, {'vf_fc0': 64, 'vf_fc1': 64}, 64, 1]}]

        # # Funções de ativação correspondentes às camadas ['pi','vf']
        # activation_functions = {
        #     'pi_fc0': nn.Tanh(),
        #     'pi_fc1': nn.Tanh(),
        #     'vf_fc0': nn.Tanh(),
        #     'vf_fc1': nn.Tanh()
        # }
        # #activation_functions = nn.ReLU()
        
        # # Transforme o dicionário em uma lista para o modelo aceitar
        # activation_functions_list = [activation_functions[key] for key in activation_functions]

        #essa linha a vi na chamada do PPO
        # policy_kwargs={
        #                 "net_arch": net_arch,
        #                 "activation_fn":activation_functions_list
        #                 },

        # Parâmetros
        batch_size = 20480  # Tamanho do lote
        n_steps = 16  # Número de passos para coletar antes de atualizar a política
        ent_coef = 0.2  # Coeficiente de entropia para incentivar a exploração
        gamma = 0.99  # Fator de desconto
        learning_rate = 2.5e-4  # Taxa de aprendizado
        clip_range = 0.2  # Limite para a razão de importância truncada (clipagem)

        model = PPO("MlpPolicy", env_p, verbose=1,
                    ent_coef=ent_coef,
                    gamma=gamma,
                    # learning_rate=learning_rate,
                    clip_range=clip_range,
                    device="cuda", batch_size=batch_size,
                    n_epochs=64,
                    n_steps=n_steps, #batch_size * num_epochs
                    tensorboard_log="./yard_tensorboard/")
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
    #env = CustomWrapper(env_p) #PreMarshEnv(render_mode='console'))
    # Executa previsões com o modelo carregado
    obs, info = env_p.reset()
    #print(obs)
    env_p.render()

    if (avalia==True) : 
        mean_reward, std_reward = evaluate_policy(model, env_p, render=False, n_eval_episodes=10)
        print(f"{tipo}: - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    else:
        i = 0
        while i < 3:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env_p.step(int(action))
            env_p.render()
            done = truncated + terminated
            print("Recompensa:", reward)
                #env.printa()
            if done:
                i +=1
                env_p.reset()
                print(env_p.dones)
                print("fim")
                print("---------------------------------------------")
            
    env_p.close()

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


def renderizaImagens(tipo, env_p,file_path):
    # Crie o ambiente
    #env = CustomWrapper(env_p) #PreMarshEnv(render_mode='console'))

    # Defina aqui o diretório onde as imagens serão salvas
    output_directory = 'output_images'

    # Crie o diretório de saída se ele não existir
    os.makedirs(output_directory, exist_ok=True)

    for ep_i in range(10):
        # Renderiza o episódio e obtém a imagem concatenada
        episode_image = render_episode(env_p, tipo,file_path)
        #episode_image= np.rot90(episode_image, k=1)
        seed = ep_i
        # Salva a imagem em um arquivo
        img_filename = os.path.join(output_directory, f"episode_image_{tipo}{ep_i}.png")
        pygame.image.save(pygame.surfarray.make_surface(episode_image), img_filename)

    # Fecha o ambiente
    env_p.close()

tipo = "PPO" #QRDQN #dqn #PPO                                                                 
#default_occupancy=0.5
num_stacks = 5
stack_height = 5
objective_size= 3
optimal_solution = ((int(stack_height/1)+stack_height) * objective_size)
max_episode_steps = optimal_solution + 1
max_episode_steps = 100
total_timesteps = 50000
discreeteAction=True

# Parâmetros a serem passados para make_env
env_params_list = [
    {
    'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': discreeteAction, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    },
    {
     'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    },
    {
    'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    },
    {
    'num_stacks': num_stacks, 'stack_height': stack_height,  'discreeteAction': True, 'max_episode_steps': max_episode_steps, 'objective_size': objective_size, 'render_mode': 'console'
    }
]



file_path=f"C:/temporario/modeloIA/yard_model_capacity_{objective_size}objetivos_tamanhoDopatio_{num_stacks}x{stack_height}"

envOriginal = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=discreeteAction, max_episode_steps=max_episode_steps, objective_size=objective_size, render_mode='console')
envWrap = CustomWrapper(envOriginal)
# Crie o ambiente vetorizado usando make_vec_env
#env_vec = make_vec_env(make_env, n_envs=len(env_params_list), env_kwargs=env_params_list)
# Crie o ambiente vetorizado usando make_vec_env
#env_vec = ([lambda: make_env(env_params) for env_params in env_params_list])
#env = CustomActionWrapper(envOriginal)

# Create the vectorized environment
#vec_env = SubprocVecEnv([make_env(i) for i in range(8)])
#vec_env = DummyVecEnv([lambda: Monitor(create_custom_env(env_params), "./yard_tensorboard") for env_params in env_params_list])

treina(total_timesteps, tipo, env_p=envWrap, file_path=file_path)


#env = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=discreeteAction, max_episode_steps=max_episode_steps, objective_size=objective_size, render_mode='console')

#carrega(tipo=tipo, avalia=True, env_p=env, file_path=file_path)
eenvOriginalRender = PreMarshEnv(num_stacks=num_stacks, stack_height=stack_height, discreeteAction=discreeteAction, max_episode_steps=max_episode_steps,  objective_size=objective_size, render_mode='rgb_array')
envWrapRender = CustomWrapper(eenvOriginalRender)
renderizaImagens(tipo, env_p=envWrapRender,file_path=file_path)
