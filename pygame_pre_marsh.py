import time
from pre_marsh_env import PreMarshEnv


# env = PreMarshEnv(10,11,False,1)
# env.reset(1,0.5,[1, 15])

#16x8 -> ....
#32x8
#64x8

env = PreMarshEnv(4,4,False, 10, 3)
env.render_mode='human'
SEED = 42
i = SEED
while i<=SEED:
    env.reset(seed=i)
    print(env.state)
    env.render()
    print(env.distancia)

    i+=1