import time
from pre_marsh_env import PreMarshEnv


# env = PreMarshEnv(10,11,False,1)
# env.reset(1,0.5,[1, 15])
env = PreMarshEnv(8,8,False)
env.render_mode='human'
SEED = 42
i = SEED
while i<=SEED:
    env.reset(seed=i)
    env.render()
    i+=1