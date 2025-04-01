import numpy as np
import pandas as pd
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, memory_size=2000):
        """
        DQN Agent for a trading environment.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Build Q-network
        self.model = self._build_model(learning_rate)
        
    def _build_model(self, learning_rate):
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))  # 3 actions
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """
        Experience replay: sample a minibatch and train the network.
        """
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, -1)
            next_state = np.array(next_state).reshape(1, -1)

            target = self.model.predict(state, verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                # Bellman equation
                t = self.model.predict(next_state, verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            states.append(state[0])
            targets.append(target)
        
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
