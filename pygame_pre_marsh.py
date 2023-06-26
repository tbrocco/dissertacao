import time
from pre_marsh_env import PreMarshEnv


# env = PreMarshEnv(10,11,False,1)
# env.reset(1,0.5,[1, 15])
env = PreMarshEnv(8,8,False)
env.reset()
env.render(mode='human')
