import torch
import gym
from src.dqn import Agent
import matplotlib.pyplot as plt
from render_browser import render_browser
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

@render_browser
def policy():
  env = gym_super_mario_bros.make('SuperMarioBros-v0')
  env = gym.wrappers.ResizeObservation(env, (224, 224))

  env = JoypadSpace(env, SIMPLE_MOVEMENT)
  obs = env.reset()
  replay_mem = 1000
  agent = Agent(obs.shape, env.action_space.n, 128, replay_mem)
  train_freq = 1000
  i = 0
  while True:
    i+=1
    step_counter = 0
    done = False  
    env.reset()
    while not done:
      yield env.render(mode='rgb_array')
      step_counter += 1
      action = agent.get_action(obs)
      next_obs, reward, done, _ = env.step(int(action))
      agent.store_env(obs, reward, next_obs, action, done)
      obs = next_obs
      if step_counter % train_freq == 0:
        agent.learn(i)
        agent.target_network.load_state_dict(agent.qNet.state_dict())


policy()