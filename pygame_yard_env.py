from yard_env import YardEnv
from yard_env import YardRenderer

env = YardEnv(10,1,11)
env.reset()
renderer = YardRenderer(env)
renderer.render_with_id()

# Verifica a observação retornada
print("seed", env.seed)
    
# Realiza algumas ações e verifica as observações e recompensas retornadas
for _ in range(10):
    # Escolhe uma ação aleatória
    action = env.action_space.sample()
    
    # Realiza a ação
    obs, reward, done, _ = env.step(action)
    
    # Verifica a observação retornada
    print("Observação:", obs)
    print("action:", action)
    # Verifica a recompensa retornada
    print("Recompensa:", reward)
    renderer.render_with_id()
    # Verifica se o jogo acabou
    if done:
        print("Jogo acabou!")
        break
