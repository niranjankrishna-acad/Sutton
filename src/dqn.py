import torch
import torch.nn as nn
import torch.optim as optim
from src.cnn import CNN
from collections import deque
import random
import numpy as np

device = torch.device("cpu")
class Agent:
  def __init__(self, input_shape, action_space, batch_size, replay_buffer_length, gamma=0.99, learning_rate=1e-3, epsilon=0.9):
    self.qNet = CNN(input_shape, action_space)
    self.target_network = CNN(input_shape, action_space)
    self.action_space = action_space
    self.batch_size = batch_size
    self.replay_buffer = deque([],maxlen=replay_buffer_length)
    self.gamma = gamma
    self.loss_fn = nn.MSELoss()
    self.optimizer = optim.Adam(self.qNet.parameters(), lr=learning_rate)
    self.epsilon = epsilon
  def get_action(self,obs):
    if np.random.randn() >= self.epsilon:
      return random.randint(0, self.action_space - 1)
    obs = torch.Tensor(obs).to(device)
    obs = obs.reshape(1,*obs.shape)
    logits = self.qNet.forward(obs).squeeze(1)
    actions = torch.argmax(logits[0], dim=0).cpu().numpy()
    return actions

  def store_env(self, obs, reward, next_obs, action, done):
    self.replay_buffer.append((obs, reward, next_obs, action, done))

  def learn(self, episode):
    data = random.sample(self.replay_buffer, self.batch_size)
    obsl = []
    rewardl = []
    next_obsl = []
    actionl = []
    donel = []
    for data_point in data:
      obs, reward, next_obs, action, done = data_point
      obsl.append(obs)
      rewardl.append(reward)
      next_obsl.append(next_obs)
      actionl.append(int(action))
      donel.append(done)

    obsl = torch.FloatTensor(obsl).to(device)
    rewardl = torch.FloatTensor(rewardl).to(device)
    next_obsl = torch.FloatTensor(next_obsl).to(device)
    
    actionl = torch.LongTensor(actionl).to(device)
    actionl = torch.unsqueeze(actionl,1)
    donel = torch.LongTensor(donel).to(device)
    with torch.no_grad():
        target_max, _ = self.target_network.forward(next_obsl).max(dim=1)
        td_target = rewardl.flatten() + self.gamma * target_max * (1 - donel.flatten())
    old_val = self.qNet.forward(obsl).gather(1, actionl).squeeze()
    loss = self.loss_fn(td_target, old_val)

    # optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()