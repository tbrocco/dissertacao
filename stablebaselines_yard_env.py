import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from yard_env import YardEnv

def treina(passos):
    env = YardEnv(10,11,True)
    env.reset()
    # Cria um ambiente vetorizado com 4 ambientes paralelos
    env = make_vec_env(lambda:  YardEnv(10,11,True), n_envs=4)

    # Define o modelo DQN
    model = DQN('MlpPolicy',env, 
                learning_rate=1e-3, buffer_size=10000, batch_size=64, 
                learning_starts=1000, train_freq=4, target_update_interval=1000, 
                exploration_fraction=0.1, exploration_final_eps=0.02, verbose=1,
                device='cuda')

    # Cria um callback para salvar o modelo a cada 10000 steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./dqn_ckpt', name_prefix='dqn')

    # Treina o modelo por 1000000 de steps
    model.learn(total_timesteps=passos, callback=checkpoint_callback, progress_bar=True)

    model.save("C:/temporario/modeloIA/dqn-gpu1")

    # Avalia o modelo
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    #print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def carrega():
    # Carrega o modelo treinado
    model = DQN.load("C:/temporario/modeloIA/dqn-gpu1")

    # Cria o ambiente
    env = YardEnv(10,11,True)

    # Executa previsões com o modelo carregado
    obs = env.reset(1,0.5)
    env.render(mode='console')
    i = 0
    while i < 10:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        env.render(mode='console')
        if done:
            break
        i +=1

    env.close()


treina(100000)
carrega()
