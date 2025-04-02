import * as React from "react"
export const About: React.FC = () => {
  const containerStyle: React.CSSProperties = {
    padding: "40px 20px",
    fontFamily: "Arial, sans-serif",
    background: "#f0f4f8",
    color: "black",
    maxWidth: "1000px",
    margin: "0 auto",
    borderRadius: "12px"
  };

  const sectionStyle: React.CSSProperties = {
    marginBottom: "30px",
    lineHeight: 1.6,
    fontSize: "1.2rem"
  };

  const headerStyle: React.CSSProperties = {
    textAlign: "center",
    marginBottom: "20px"
  };
  return (
    <div style={containerStyle}>
    <h1 style={headerStyle}>About Our Trading Platform</h1>
    <div style={sectionStyle}>
      <p>
        Our trading platform uses advanced machine learning models—LSTM, XGBoost, NLP sentiment analysis, and Reinforcement Learning—to provide reliable, data-driven stock predictions.
      </p>
      <p>
        Whether you're a seasoned trader or a newcomer, our platform is designed to simplify decision-making and boost confidence through transparency and predictive insight.
      </p>
    </div>

    <div style={sectionStyle}>
      <h3>Team</h3>
      <p style={{ textAlign: "center", marginTop: "10px" }}>
        Gabriel Brown, Yagiz Idilamn, Odilon Quevillon, Adian Pawlowski
      </p>
    </div>
  </div>
  );
}
