import gymnasium
from pre_marsh_env import PreMarshEnv
import numpy as np
from gymnasium.utils import play 
#check
from gymnasium.utils.env_checker import check_env
from gymnasium import spaces
from gymnasium.wrappers import monitoring
#from gymnasium.wrappers.monitoring import load_results


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

# Cria uma instância do ambiente
env = CustomWrapper(PreMarshEnv())
#env = PreMarshEnv()

def check():
    #check
    check_env(env)

def doubleCheck():
        
    env = PreMarshEnv(16,8,False, None, 16, 5)
    env.render_mode='console'
    SEED = 42
    # Verifica o espaço de ação
    print("Espaço de ação:", env.action_space)
    # Verifica o espaço de observação
    print("Espaço de observação:", env.observation_size)
    # Verifica o espaço de observação
    print("Espaço de observação:", env.observation_space)
    print(env.observation_size)
    # Reinicia o ambiente
    # state = env.reset(333, 0.5,[55, 41, 38, 32, 33] )
    state, info = env.reset(333 )
    print(env.observation_size)
    print("ac: ", env.default_occupancy)
    state = env.stateDictToArray(env.state)
    print(len(state))
    print(env.state)
    #play.play(env, zoom=3, keys_to_action={"m": np.array([0,0,0])})

    # Realiza algumas ações e verifica as observações e recompensas retornadas
    reward_sum = 0
    acoes = []
    rewards = []
    for _ in range(20):
        # Escolhe uma ação aleatória
        action = env.action_space.sample()
        
        # Realiza a ação
        obs, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        # Verifica a observação retornada
        #print("Observação:", obs)
        
        # Verifica a recompensa retornada
        # print("Objetivo:", env.objective)
        # print("curr-move", env.current_action)
        # print("curr-pilhas_quantidade_placas", env.pilhas_quantidade_placas)
        # print("curr-pilhas_quantidade_placas_do_objetivo", env.pilhas_quantidade_placas_do_objetivo)
        # print("curr-pilhas_quantidade_placas_do_objetivo_desbloqueadas", env.pilhas_quantidade_placas_do_objetivo_desbloqueadas)
        # print("pilhas_distancia_placas_do_objetivo", env.pilhas_distancia_placas_do_objetivo)
        # print("curr-quantidade_placas_do_objetivo_desbloqueadas", env.quantidade_placas_do_objetivo_desbloqueadas)
        env.render()
        state = env.stateDictToArray(obs)
        print(reward)
        print(state)
        acoes.append(action)
        rewards.append(reward)
        done = truncated + terminated
        if done == True:
            print("Recompensa:", reward, reward_sum)
            reward_sum = 0
            env.reset()

    print("Recompensa:", reward, reward_sum)
    print("Acoes:", acoes)
    print("Recompensas:",  rewards)
# def grava_video():
#     # envolve o ambiente com o Monitor
#     gym.logger.set_level(gym.logger.DEBUG)
#     envM = Monitor(env, write_upon_reset=True, directory='videos', force=True,video_callable=lambda episode_id: True )
#     obs = envM.reset()
#     envM.render(mode='rgb_array')
#     # interage com o ambiente por 1000 passos
#     for i in range(20):
#         action = envM.action_space.sample()
#         obs, reward, done, info = envM.step(action)
#         envM.render(mode='rgb_array')
#         if done:
#             obs = envM.reset()

#     # fecha o ambiente e salva o vídeo
#     envM.close()



# def toca_video(): 
#     # Carregar o arquivo de resultados do Monitor Wrapper
#     results = load_results('videos')

#check()
doubleCheck()
#grava_video()
#toca_video()

