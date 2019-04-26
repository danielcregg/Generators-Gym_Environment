from setuptools import setup

setup(name='gym_generators',
      version='0.0.1',
      keywords='environment, agent, rl, openaigym, openai-gym, gym',
      url='https://github.com/danielcregg/gym-generators',
      description='Generators environment package for OpenAI Gym',
      packages=['gym_generators', 'gym_generators.envs'],
      install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0',
        'pandas'
      ]
)  
