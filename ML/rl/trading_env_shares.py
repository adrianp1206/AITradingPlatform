import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math

class StockTradingEnv:
    def __init__(self, df, initial_balance=10000):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        
        # Define actions: 0: hold, 1: long, 2: short.
        self.action_space = [0, 1, 2]
        # State includes 7 features plus the position indicator (-1 for short, 0 for flat, 1 for long)
        self.state_size = 15 + 1
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = 0.0  # Price at which the current position was opened
        self.total_shares = 0   # Number of shares currently held or shorted
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

        # Action 1: Go Long
        if action == 1:
            if self.position == 0:
                # Use full balance to buy as many shares as possible
                self.total_shares = math.floor(self.balance / current_price)
                if self.total_shares > 0:
                    self.position = 1
                    self.entry_price = current_price
                    # Deduct used funds (optionally, you may leave residual cash if desired)
                    self.balance -= self.total_shares * current_price
            elif self.position == -1:
                # Close short position first: profit = (entry_price - current_price) * shares
                pnl = (self.entry_price - current_price) * self.total_shares
                self.balance += self.total_shares * self.entry_price + pnl  # cover the short and add profit
                self.realized_profit += pnl
                reward = pnl
                # Now open a long position using the full updated balance
                self.position = 0  # reset position before opening long
                self.total_shares = 0
                # Then recursively open long (could also inline the logic)
                available_balance = self.balance
                self.total_shares = math.floor(available_balance / current_price)
                if self.total_shares > 0:
                    self.position = 1
                    self.entry_price = current_price
                    self.balance -= self.total_shares * current_price
            # If already long, do nothing

        # Action 2: Go Short
        elif action == 2:
            if self.position == 0:
                # Use full balance to short as many shares as possible.
                # For simplicity, assume you can short shares equal in value to your current balance.
                self.total_shares = math.floor(self.balance / current_price)
                if self.total_shares > 0:
                    self.position = -1
                    self.entry_price = current_price
                    # In a real environment, you might not deduct balance immediately for a short,
                    # but for simulation, we can assume the full balance is allocated.
            elif self.position == 1:
                # Close long position first: profit = (current_price - entry_price) * shares
                pnl = (current_price - self.entry_price) * self.total_shares
                self.balance += self.total_shares * current_price + pnl  # sell long and add profit
                self.realized_profit += pnl
                reward = pnl
                # Now open a short position using the full updated balance
                self.position = 0  # reset position before opening short
                self.total_shares = 0
                available_balance = self.balance
                self.total_shares = math.floor(available_balance / current_price)
                if self.total_shares > 0:
                    self.position = -1
                    self.entry_price = current_price
            # If already short, do nothing

        # Action 0: Hold => do nothing

        # Move to next time step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            # At episode end, close any open position at current price.
            if self.position != 0:
                final_price = current_price  # using current price as final price
                if self.position == 1:
                    pnl = (final_price - self.entry_price) * self.total_shares
                elif self.position == -1:
                    pnl = (self.entry_price - final_price) * self.total_shares
                self.balance += self.total_shares * final_price + pnl  # realize final proceeds
                self.realized_profit += pnl
                reward += pnl
                self.position = 0
                self.total_shares = 0
                self.entry_price = 0.0
            done = True
        
        next_state = self._get_state()
        info = {
            "realized_profit": self.realized_profit,
            "balance": self.balance,
            "total_shares": self.total_shares,
            "position": self.position
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
            current_data['LSTM_Pred'],
            current_data['Sentiment_Score'],
            current_data['DE Ratio'],
            current_data['Return on Equity'],
            current_data['Price/Book'],
            current_data['Profit Margin'],
            current_data['Diluted EPS'],
            current_data['Beta'],
            self.position  # -1/0/1 or 0/1 depending on the environment
        ]
        return np.array(state, dtype=np.float32)


import numpy as np
import math

class StockTradingEnvLongOnly:
    def __init__(self, df, initial_balance=10000):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance

        # Actions: 0 = Hold, 1 = Buy (Long), 2 = Sell (only if holding)
        self.action_space = [0, 1, 2]
        self.state_size = 15 + 1  # 7 features + 1 position
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # 0 = no position, 1 = long
        self.entry_price = 0.0
        self.total_shares = 0
        self.balance = self.initial_balance
        self.realized_profit = 0.0
        return self._get_state()

    def step(self, action):
        reward = 0
        done = False

        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        if action == 1 and self.position == 0:
            # Open long position
            self.total_shares = math.floor(self.balance / current_price)
            if self.total_shares > 0:
                self.entry_price = current_price
                self.balance -= self.total_shares * current_price
                self.position = 1

        elif action == 2 and self.position == 1:
            # Close long position
            pnl = (current_price - self.entry_price) * self.total_shares
            self.balance += self.total_shares * current_price + pnl
            self.realized_profit += pnl
            reward = pnl
            self.position = 0
            self.total_shares = 0
            self.entry_price = 0.0

        # Action 0: Hold - do nothing

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            if self.position == 1:
                # Close position at the end
                final_price = current_price
                pnl = (final_price - self.entry_price) * self.total_shares
                self.balance += self.total_shares * final_price + pnl
                self.realized_profit += pnl
                reward += pnl
                self.position = 0
                self.total_shares = 0
                self.entry_price = 0.0
            done = True

        next_state = self._get_state()
        info = {
            "realized_profit": self.realized_profit,
            "balance": self.balance,
            "total_shares": self.total_shares,
            "position": self.position
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
            current_data['LSTM_Pred'],
            current_data['Sentiment_Score'],
            current_data['DE Ratio'],
            current_data['Return on Equity'],
            current_data['Price/Book'],
            current_data['Profit Margin'],
            current_data['Diluted EPS'],
            current_data['Beta'],
            self.position  # -1/0/1 or 0/1 depending on the environment
        ]
        return np.array(state, dtype=np.float32)
