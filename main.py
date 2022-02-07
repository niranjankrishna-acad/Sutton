import torch
import gym
from PIL import Image
from src.dqn import Agent
import matplotlib.pyplot as plt
from render_browser import render_browser
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst
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
      cam = GradCAM(model=agent.qNet, target_layers=agent.qNet.networkConv, use_cuda=False)

      
      env.render(mode='rgb_array')
      step_counter += 1
      action = agent.get_action(obs)
      next_obs, reward, done, _ = env.step(int(action))
      grayscale_cam = cam(input_tensor=torch.Tensor(next_obs).unsqueeze(0))
      visualization = show_cam_on_image(Image.new(mode="RGB", size=(224,224)), grayscale_cam, use_rgb=True)
      
      yield np.hstack((next_obs, next_obs + visualization))
      agent.store_env(obs, reward, next_obs, action, done)
      obs = next_obs
      if step_counter % train_freq == 0:
        agent.learn(i)
        agent.target_network.load_state_dict(agent.qNet.state_dict())


policy()