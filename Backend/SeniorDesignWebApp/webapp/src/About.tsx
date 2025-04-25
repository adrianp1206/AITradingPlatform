import * as React from "react";

export const About: React.FC = () => {
  return (
    <div style={{
      width: "100%",
      backgroundColor: "#f7fafc",
      fontFamily: "Arial, sans-serif",
      color: "#333",
      padding: "20px",
      boxSizing: "border-box",
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "flex-start" 
    }}>

      <h1 style={{
        fontSize: "2.8rem",
        fontWeight: "700",
        marginTop: "20px",  
        marginBottom: "20px",
        textAlign: "center",
        color: "#1a202c"
      }}>
        About Our Trading Platform
      </h1>

      <div style={{
        maxWidth: "100%",
        fontSize: "1.1rem",
        lineHeight: "1.8",
        textAlign: "center",
        marginBottom: "40px",
        padding: "0 20px"
      }}>
        <p>
          Our trading platform leverages state-of-the-art machine learning technologies—including Long Short-Term Memory (LSTM), XGBoost, Natural Language Processing (NLP) sentiment analysis, and Reinforcement Learning—to deliver accurate, data-driven stock predictions.
        </p>
        <p style={{ marginTop: "20px" }}>
          Designed for both seasoned traders and newcomers, our platform emphasizes transparency, ease-of-use, and actionable insights, enabling users to make informed trading decisions with confidence.
        </p>
      </div>

      <div style={{ textAlign: "center" }}>
        <h3 style={{ fontSize: "2rem", marginBottom: "10px", color: "#2d3748" }}>Meet the Team</h3>
        <p style={{ fontSize: "1.1rem", color: "#4a5568" }}>
          Gabriel Brown • Yagiz Idilamn • Odilon Quevillon • Adian Pawlowski
        </p>
      </div>
    </div>
  );
};

