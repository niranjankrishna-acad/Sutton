while not done:
    env.render()
    step_counter += 1
    action = agent.get_action(obs)
    next_obs, reward, done, _ = env.step(action)
    agent.store_env(obs, reward, next_obs, action, done)
    obs = next_obs
    if step_counter % train_freq == 0:
      agent.learn(i + 1)
      agent.target_network.load_state_dict(agent.qNet.state_dict())