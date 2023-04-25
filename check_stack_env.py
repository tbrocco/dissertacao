import gym
from pre_marsh_env import PreMarshEnv
import numpy as np
from gym.utils import play 
#check
from gym.utils.env_checker import check_env
from gym import spaces
from gym.wrappers import Monitor
from gym.wrappers.monitor import load_results


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

# Cria uma instância do ambiente
env = CustomWrapper(PreMarshEnv())
env = PreMarshEnv()
env.reset()

def check():
    #check
    check_env(env)

def doubleCheck():
    # Verifica o espaço de ação
    print("Espaço de ação:", env.action_space)
    # Verifica o espaço de observação
    print("Espaço de observação:", env.observation_size)
    # Verifica o espaço de observação
    print("Espaço de observação:", env.observation_space)

    # Reinicia o ambiente
    env.reset(1, 0.5,[55, 41, 38, 32, 33] )
    env.render_mode = "rgb_array"
    env.render_mode = "human"
    print(env.state)
    #play.play(env, zoom=3, keys_to_action={"m": np.array([0,0,0])})

    # Realiza algumas ações e verifica as observações e recompensas retornadas
    for _ in range(10):
        # Escolhe uma ação aleatória
        action = env.action_space.sample()
        
        # Realiza a ação
        obs, reward, done, _ = env.step(action)
        
        # Verifica a observação retornada
        print("Observação:", obs)
        
        # Verifica a recompensa retornada
        print("Recompensa:", reward)
        print("Objetivo:", env.objective)
        print("curr-move", env.current_action)
        print("Done:", done)
        print("curr-pilhas_quantidade_placas", env.pilhas_quantidade_placas)
        print("curr-pilhas_quantidade_placas_do_objetivo", env.pilhas_quantidade_placas_do_objetivo)
        print("curr-pilhas_quantidade_placas_do_objetivo_desbloqueadas", env.pilhas_quantidade_placas_do_objetivo_desbloqueadas)
        print("pilhas_distancia_placas_do_objetivo", env.pilhas_distancia_placas_do_objetivo)
        print("curr-quantidade_placas_do_objetivo_desbloqueadas", env.quantidade_placas_do_objetivo_desbloqueadas)
        state = env.stateDictToArray(obs)
        print(state)
        print(env.observation_size)
        print(len(state))
        if done == True:
            env.reset()


def grava_video():
    # envolve o ambiente com o Monitor
    gym.logger.set_level(gym.logger.DEBUG)
    envM = Monitor(env, write_upon_reset=True, directory='videos', force=True,video_callable=lambda episode_id: True )
    obs = envM.reset()
    envM.render(mode='rgb_array')
    # interage com o ambiente por 1000 passos
    for i in range(20):
        action = envM.action_space.sample()
        obs, reward, done, info = envM.step(action)
        envM.render(mode='rgb_array')
        if done:
            obs = envM.reset()

    # fecha o ambiente e salva o vídeo
    envM.close()



def toca_video(): 
    # Carregar o arquivo de resultados do Monitor Wrapper
    results = load_results('videos')

#check()
doubleCheck()
#grava_video()
#toca_video()

