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
        
        # Define actions: 0: hold, 1: long, 2: short.
        self.action_space = [0, 1, 2]
        # The state now includes 7 features plus position indicator (-1, 0, or 1)
        self.state_size = 7 + 1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0.0  # Price at which the current position was opened
        self.balance = self.initial_balance
        self.realized_profit = 0.0
        return self._get_state()

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Actions:
          0: Hold
          1: Go long (or switch from short to long)
          2: Go short (or switch from long to short)
        """
        reward = 0
        done = False
        
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        # Action 1: Long
        if action == 1:
            if self.position == 0:  # Open a long position
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:  # Close short and open long
                pnl = (self.entry_price - current_price)  # Profit from short: entry - current
                self.balance += pnl
                self.realized_profit += pnl
                reward = pnl
                # Open long position at current price
                self.position = 1
                self.entry_price = current_price
            # if already long, do nothing

        # Action 2: Short
        elif action == 2:
            if self.position == 0:  # Open a short position
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:  # Close long and open short
                pnl = (current_price - self.entry_price)  # Profit from long: current - entry
                self.balance += pnl
                self.realized_profit += pnl
                reward = pnl
                # Open short position at current price
                self.position = -1
                self.entry_price = current_price
            # if already short, do nothing

        # Action 0: Hold => do nothing

        # Move to the next time step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            # Optionally close any open position at the end of the episode
            if self.position != 0:
                final_price = current_price  # using current_price as final price
                if self.position == 1:
                    pnl = (final_price - self.entry_price)
                elif self.position == -1:
                    pnl = (self.entry_price - final_price)
                self.balance += pnl
                self.realized_profit += pnl
                reward += pnl
                self.position = 0
                self.entry_price = 0.0
            done = True
        
        next_state = self._get_state()
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
            self.position  # -1 for short, 0 for flat, 1 for long
        ]
        return np.array(state, dtype=np.float32)
