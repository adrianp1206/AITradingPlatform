import pandas as pd

# Build a list of dictionaries, each representing one prediction row.

data_rows = [
    # KO data
    {
        "ticker": "KO",
        "lstm_predicted_price": 70.39,  # April 2nd prediction
        "next_trading_date": "2025-04-02",
        "nlp_sentiment_score": 0.05,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-01T22:30:11.415784+00:00",
        "xgboost_prob": 0.99,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "KO",
        "lstm_predicted_price": 70.63991220653057,  # April 3rd
        "next_trading_date": "2025-04-03",
        "nlp_sentiment_score": 0.04280413151666463,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-02T22:30:11.415784+00:00",
        "xgboost_prob": 0.3288005292415619,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "KO",
        "lstm_predicted_price": 70.9596065825224,  # April 4th
        "next_trading_date": "2025-04-04",
        "nlp_sentiment_score": 0.03712235464181992,
        "rl_recommendation": "Buy",
        "updated_at": "2025-04-03T22:30:12.786631+00:00",
        "xgboost_prob": 0.923660933971405,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "KO",
        "lstm_predicted_price": 70.67864154279232,  # April 8th
        "next_trading_date": "2025-04-08",
        "nlp_sentiment_score": 0.12997203827932066,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-08T03:37:47.250199+00:00",
        "xgboost_prob": 0.20819835364818573,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "KO",
        "lstm_predicted_price": 70.03479933202267,  # April 9th
        "next_trading_date": "2025-04-09",
        "nlp_sentiment_score": 0.11601663022696365,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-08T23:28:32.381848+00:00",
        "xgboost_prob": 0.7025121450424194,
        "xgboost_signal": "Up"
    },
    # TSLA data
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 255.04,  # April 2nd prediction
        "next_trading_date": "2025-04-02",
        "nlp_sentiment_score": -0.19,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-01T22:30:17.619640+00:00",
        "xgboost_prob": 0.74,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 260.5894725187123,  # April 3rd
        "next_trading_date": "2025-04-03",
        "nlp_sentiment_score": -0.1365712463112854,
        "rl_recommendation": "Buy",
        "updated_at": "2025-04-02T22:30:17.619640+00:00",
        "xgboost_prob": 0.8466350436210632,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 263.13582161039113,  # April 4th
        "next_trading_date": "2025-04-04",
        "nlp_sentiment_score": -0.17184036953675175,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-03T22:30:21.646113+00:00",
        "xgboost_prob": 0.16973721981048584,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 262.47370475903153,  # April 8th
        "next_trading_date": "2025-04-08",
        "nlp_sentiment_score": -0.23322285491384015,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-08T03:37:51.404441+00:00",
        "xgboost_prob": 0.18675316870212555,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 259.1795313115418,  # April 9th
        "next_trading_date": "2025-04-09",
        "nlp_sentiment_score": -0.12368009297616607,
        "rl_recommendation": "Buy",
        "updated_at": "2025-04-08T23:28:36.799434+00:00",
        "xgboost_prob": 0.15772251784801483,
        "xgboost_signal": "Down"
    },
    # MSFT data
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 389.27,  # April 2nd prediction
        "next_trading_date": "2025-04-02",
        "nlp_sentiment_score": -0.03,
        "rl_recommendation": "Buy",
        "updated_at": "2025-04-01T22:30:25.867709+00:00",
        "xgboost_prob": 0.94,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 387.8792772629857,  # April 3rd
        "next_trading_date": "2025-04-03",
        "nlp_sentiment_score": -0.030077809929009047,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-02T22:30:25.867709+00:00",
        "xgboost_prob": 0.6178485751152039,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 386.037409017086,  # April 4th
        "next_trading_date": "2025-04-04",
        "nlp_sentiment_score": -0.0068380181378551695,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-03T22:30:27.146125+00:00",
        "xgboost_prob": 0.12348318845033646,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 381.094131282568,  # April 8th
        "next_trading_date": "2025-04-08",
        "nlp_sentiment_score": -0.04941972919261191,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-08T03:37:55.901848+00:00",
        "xgboost_prob": 0.7890827655792236,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 378.32228784918783,  # April 9th
        "next_trading_date": "2025-04-09",
        "nlp_sentiment_score": -0.04211933627727795,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-08T23:28:41.264034+00:00",
        "xgboost_prob": 0.42910876870155334,
        "xgboost_signal": "Down"
    },
    {
        "ticker": "KO",
        "lstm_predicted_price": 69.45874240338803,  # April 9th
        "next_trading_date": "2025-04-10",
        "nlp_sentiment_score": 0.1048747821741144,
        "rl_recommendation": "Hold",
        "updated_at": "2025-04-09T23:19:16.264034+00:00",
        "xgboost_prob": 0.7586953043937683,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "TSLA",
        "lstm_predicted_price": 250.4057342903316,  # April 9th
        "next_trading_date": "2025-04-10",
        "nlp_sentiment_score": -0.15365738051452674,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-09T23:28:41.264034+00:00",
        "xgboost_prob": 0.9813609719276428,
        "xgboost_signal": "Up"
    },
    {
        "ticker": "MSFT",
        "lstm_predicted_price": 375.582495508194,  # April 9th
        "next_trading_date": "2025-04-10",
        "nlp_sentiment_score": -0.0557100507034495,
        "rl_recommendation": "Sell",
        "updated_at": "2025-04-09T23:28:41.264034+00:00",
        "xgboost_prob": 0.989266037940979,
        "xgboost_signal": "Up"
    }
]

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(data_rows)

# Optionally, sort the DataFrame by ticker and trading date:
df['next_trading_date'] = pd.to_datetime(df['next_trading_date'])
df = df.sort_values(by=['ticker', 'next_trading_date'])

# Write the DataFrame to a CSV file
csv_filename = "predictions.csv"
df.to_csv(csv_filename, index=False)

print(f"{csv_filename} created with {len(df)} rows.")
