import * as React from "react";
import { useLoaderData } from 'react-router-dom';

interface PredictionData {
  ticker: string;
  lstm_predicted_price: number;
  nlp_sentiment_score: number;
  rl_recommendation: string;
  updated_at: string;
  xgboost_signal: string;
}

export const Predictions: React.FC = () => {
  const predictions = useLoaderData() as PredictionData[];

  const getColor = (value: string | number) => {
    if (typeof value === 'number') {
      return value > 0.5 ? 'green' : 'red';
    }
    return value === "Buy" || value === "Up" ? 'green' : 'red';
  };

  const containerStyle: React.CSSProperties = {
    padding: "2rem",
    fontFamily: "Montserrat, sans-serif",
    background: "#f0f4f8",
    borderRadius: "12px",
    color: "#000",
    maxWidth: "1200px",
    margin: "2rem auto",
    boxSizing: "border-box"
  };

  const cardStyle: React.CSSProperties = {
    background: "#ffffff",
    borderRadius: "8px",
    padding: "24px 16px",
    margin: "30px 0",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
    display: "flex",
    justifyContent: "space-around",
    alignItems: "center",
    flexWrap: "wrap",
    width: "100%",
    boxSizing: "border-box"
  };

  const cardSectionStyle: React.CSSProperties = {
    flex: "1",
    minWidth: "180px",
    margin: "12px",
    textAlign: "center"
  };

  return (
    <div style={containerStyle}>
      <h2 style={{ textAlign: "center", fontSize: "2rem", marginBottom: "2rem" }}>
        Today's Recommendations
      </h2>
      {predictions.map((prediction) => (
        <div key={prediction.ticker} style={cardStyle}>
          <div style={cardSectionStyle}><h3>{prediction.ticker}</h3></div>
          <div style={cardSectionStyle}>
            <strong>LSTM Predicted Price:</strong><br />
            ${prediction.lstm_predicted_price.toFixed(2)}
          </div>
          <div style={cardSectionStyle}>
            <strong>XGBoost Signal:</strong><br />
            <span style={{ color: getColor(prediction.xgboost_signal) }}>{prediction.xgboost_signal}</span>
          </div>
          <div style={cardSectionStyle}>
            <strong>NLP Sentiment Score:</strong><br />
            <span style={{ color: getColor(prediction.nlp_sentiment_score) }}>{prediction.nlp_sentiment_score.toFixed(2)}</span>
          </div>
          <div style={cardSectionStyle}>
            <strong>Reinforcement Learning:</strong><br />
            <span style={{ color: getColor(prediction.rl_recommendation) }}>{prediction.rl_recommendation}</span>
          </div>
          <div style={cardSectionStyle}>
            <em style={{ fontSize: "12px", color: "#777" }}>
              Last Updated: {new Date(prediction.updated_at).toLocaleString()}
            </em>
          </div>
        </div>
      ))}
    </div>
  );
};