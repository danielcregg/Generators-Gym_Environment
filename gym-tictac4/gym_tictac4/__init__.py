from gym.envs.registration import register

register(
    id='generators-v0',
    entry_point='gym_generators.envs:Generators',
)
