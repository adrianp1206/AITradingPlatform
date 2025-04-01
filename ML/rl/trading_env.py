import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class StockTradingEnv:
    def __init__(self, df, initial_balance=10000):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        
        self.action_space = [0, 1, 2]  # hold, buy, sell
        self.state_size = 7 + 1       # 7 features + position

        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.total_shares = 0
        self.last_price = 0
        self.balance = self.initial_balance
        
        # NEW: Track only realized profit
        self.realized_profit = 0.0

        return self._get_state()

    def step(self, action):
        """
        Returns: (next_state, reward, done, info)
        info can include any debugging or logging variables you like.
        """
        reward = 0
        done = False
        
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        if action == 1:  # buy
            if self.position == 0:
                self.position = 1
                self.total_shares = 1
                self.last_price = current_price

        elif action == 2:  # sell
            if self.position == 1:
                pnl = (current_price - self.last_price) * self.total_shares
                self.balance += pnl
                
                # Update environment's realized P&L tracker
                self.realized_profit += pnl

                # RL reward can still be whatever you want:
                reward = pnl  # e.g., realized profit only
                
                # Flatten position
                self.position = 0
                self.total_shares = 0

        # OPTIONAL: If you *don't* want partial unrealized reward, leave it out.
        # if self.position == 1:
        #     unrealized_pnl = (current_price - self.last_price) * self.total_shares
        #     reward += unrealized_pnl * 0.01

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
        
        next_state = self._get_state()

        # Return info with realized profit for logging
        info = {
            "realized_profit": self.realized_profit,
            "balance": self.balance,
        }
        return next_state, reward, done, info

    def _get_state(self):
        current_data = self.df.iloc[self.current_step]
        state = [
            current_data['Open'],
            current_data['High'],
            current_data['Low'],
            current_data['Close'],
            current_data['Volume'],
            current_data['XGB_Pred'],
            current_data['XGB_Prob_Up'],
            self.position
        ]
        return np.array(state, dtype=np.float32)
