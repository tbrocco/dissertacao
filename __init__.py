from gym.envs.registration import register

# register(
#     id='stack-v0',
#     entry_point='nomedoarquivo.stack_env:StackEnv',
# )
register(
    id='stack-v0',
    entry_point='stack_env.stack_env:StackEnv',
)