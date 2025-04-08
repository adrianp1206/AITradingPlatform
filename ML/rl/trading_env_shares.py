import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math

import numpy as np
import pandas as pd
import math

class StockTradingEnv:
    def __init__(self, df, initial_balance=10000):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        
        # Define actions: 
        #   0 = Hold, 
        #   1 = Go long (or switch from short to long),
        #   2 = Go short (or switch from long to short).
        self.action_space = [0, 1, 2]

        # State includes 15 features plus the position indicator (-1 for short, 0 for flat, 1 for long)
        self.state_size = 15 + 1  
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0        # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0   # Price at which the current position was opened
        self.total_shares = 0    # Number of shares currently held or shorted
        self.balance = self.initial_balance
        self.realized_profit = 0.0
        return self._get_state()

    def step(self, action):
        """
        Actions:
          0: Hold
          1: Go long (or switch from short to long)
          2: Go short (or switch from long to short)
        """
        reward = 0.0
        done = False
        
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        # -------------------------------
        # ACTION 1: GO LONG
        # -------------------------------
        if action == 1:
            if self.position == 0:
                # OPEN LONG from flat
                shares_to_buy = math.floor(self.balance / current_price)
                if shares_to_buy > 0:
                    self.position = 1
                    self.entry_price = current_price
                    self.total_shares = shares_to_buy
                    self.balance -= shares_to_buy * current_price

            elif self.position == -1:
                # SWITCH from SHORT to LONG
                # 1) Close the SHORT
                reward = self._close_short(current_price)
                # 2) Now open a LONG with available balance
                shares_to_buy = math.floor(self.balance / current_price)
                if shares_to_buy > 0:
                    self.position = 1
                    self.entry_price = current_price
                    self.total_shares = shares_to_buy
                    self.balance -= shares_to_buy * current_price

            # if already long, do nothing

        # -------------------------------
        # ACTION 2: GO SHORT
        # -------------------------------
        elif action == 2:
            if self.position == 0:
                # OPEN SHORT from flat
                shares_to_short = math.floor(self.balance / current_price)
                if shares_to_short > 0:
                    self.position = -1
                    self.entry_price = current_price
                    self.total_shares = shares_to_short
                    # Credit the proceeds of the short sale to balance
                    self.balance += shares_to_short * current_price

            elif self.position == 1:
                # SWITCH from LONG to SHORT
                # 1) Close the LONG
                reward = self._close_long(current_price)
                # 2) Open the SHORT
                shares_to_short = math.floor(self.balance / current_price)
                if shares_to_short > 0:
                    self.position = -1
                    self.entry_price = current_price
                    self.total_shares = shares_to_short
                    self.balance += shares_to_short * current_price

            # if already short, do nothing

        # -------------------------------
        # ACTION 0: HOLD
        # -------------------------------
        # (do nothing)

        # Move to the next time step
        self.current_step += 1

        # If we are at the end, close any open position
        if self.current_step >= self.n_steps - 1:
            if self.position == 1:
                # Close long
                reward += self._close_long(current_price)
            elif self.position == -1:
                # Close short
                reward += self._close_short(current_price)
            done = True

        next_state = self._get_state()
        info = {
            "realized_profit": self.realized_profit,
            "balance": self.balance,
            "total_shares": self.total_shares,
            "position": self.position
        }
        return next_state, reward, done, info

    # ----------------------------------------------------
    # HELPER: Close a long position
    # ----------------------------------------------------
    def _close_long(self, current_price):
        """
        Close out a long position at the given price.
        Returns the profit (reward).
        """
        if self.position != 1 or self.total_shares == 0:
            return 0.0  # No long to close

        # Sale proceeds
        proceeds = self.total_shares * current_price
        self.balance += proceeds

        # PnL
        pnl = (current_price - self.entry_price) * self.total_shares
        self.realized_profit += pnl

        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.total_shares = 0

        return pnl

    # ----------------------------------------------------
    # HELPER: Close a short position
    # ----------------------------------------------------
    def _close_short(self, current_price):
        """
        Close out a short position at the given price.
        Returns the profit (reward).
        """
        if self.position != -1 or self.total_shares == 0:
            return 0.0  # No short to close

        # Cost to buy back shares
        cost = self.total_shares * current_price
        self.balance -= cost

        # PnL from short
        pnl = (self.entry_price - current_price) * self.total_shares
        self.realized_profit += pnl

        # Reset position
        self.position = 0
        self.entry_price = 0.0
        self.total_shares = 0

        return pnl

    # ----------------------------------------------------
    # GET ENV STATE
    # ----------------------------------------------------
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
            float(self.position)  # -1, 0, or 1
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
        # e.g., 15 features (Open, High, Low, Close, Volume, XGB_Pred, etc.) + 1 position
        self.state_size = 15 + 1  
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
        reward = 0.0
        done = False

        current_data = self.df.iloc[self.current_step]
        current_price = current_data['Close']

        # --------------------------
        # ACTION = 1 => OPEN LONG
        # --------------------------
        if action == 1 and self.position == 0:
            # Buy as many shares as possible with current balance
            self.total_shares = math.floor(self.balance / current_price)
            if self.total_shares > 0:
                self.entry_price = current_price
                self.balance -= self.total_shares * current_price
                self.position = 1

        # --------------------------
        # ACTION = 2 => CLOSE LONG
        # --------------------------
        elif action == 2 and self.position == 1:
            pnl = (current_price - self.entry_price) * self.total_shares
            # Add *only the sale proceeds* to the balance
            self.balance += self.total_shares * current_price

            self.realized_profit += pnl
            reward = pnl

            # Reset position
            self.position = 0
            self.total_shares = 0
            self.entry_price = 0.0

        # ACTION = 0 => HOLD (do nothing)

        # --------------------------
        # NEXT STEP or END
        # --------------------------
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            # At the final step, if still long, close the position
            if self.position == 1:
                final_price = current_price
                pnl = (final_price - self.entry_price) * self.total_shares

                self.balance += self.total_shares * final_price
                self.realized_profit += pnl
                reward += pnl

                # Reset position
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
            self.position
        ]
        return np.array(state, dtype=np.float32)

