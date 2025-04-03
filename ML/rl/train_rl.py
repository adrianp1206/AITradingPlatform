

def train_dqn_agent(agent, env, episodes=10, save_path=None):
    """
    Train the DQN agent on the given StockTradingEnv.
    
    Each episode runs from start to end of the DataFrame once.
    """

    print("In training")
    for e in range(episodes):
        state = env.reset()  # reset environment
        total_reward = 0
        done = False
        print("starting loop in training")
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {e+1}/{episodes} finished. Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    if save_path:
        agent.model.save(save_path)
        print(f"Model saved to: {save_path}")
