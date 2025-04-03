import * as React from "react";

export const Home: React.FC = () => {
  return (
    <div style={{ backgroundColor: "#f0f4f8", color: "#000", fontFamily: "Montserrat, sans-serif", padding: "2rem" }}>
      <header style={{ backgroundColor: "#1c1c1e", padding: "1rem 2rem", borderRadius: "8px", marginBottom: "2rem" }}>
        <h1 style={{ color: "#fff", fontSize: "2rem", textAlign: "center" }}>
          <i className="fas fa-money-bill-trend-up" style={{ marginRight: ".5rem", color: "coral" }}></i>
          Trading Recommendation Platform
        </h1>
      </header>

      <section style={{ marginBottom: "3rem", textAlign: "center" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "1rem" }}>Want better informed decisions for trading stocks?</h2>
        <p style={{ fontSize: "1.2rem", maxWidth: "800px", margin: "0 auto" }}>
          Get recommendations for free using our app!
        </p>
      </section>

      <section style={{ marginBottom: "3rem", backgroundColor: "#fff", padding: "2rem", borderRadius: "10px" }}>
        <h2 style={{ fontSize: "1.8rem", marginBottom: "1rem" }}>What we offer:</h2>
        <p style={{ fontSize: "1.1rem" }}>
          Our platform leverages advanced predictive machine learning models to generate trade recommendations
          across a variety of stocks while illustrating the daily performance of our models. Empowering the 
          average investor to make more informed decisions.
        </p>
      </section>

      <section style={{ marginBottom: "3rem" }}>
        <h2 style={{ fontSize: "2.2rem", textAlign: "center", marginBottom: "2rem" }}>Platform Components</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.5rem" }}>
          {[
            {
              title: "Fundamental Indicators",
              text: "Metrics like ROE, PE Ratio, and ROIC are used to enhance model accuracy."
            },
            {
              title: "Technical Indicators",
              text: "We leverage Moving Averages, Sharpe Ratio, Bollinger Bands, and others to fine-tune forecasts."
            },
            {
              title: "Long Short-Term Memory",
              text: "This model examines historical stock data and predicts next-day prices."
            },
            {
              title: "XGBoost",
              text: "Evaluates a wide range of indicators to forecast stock movement probabilities."
            },
            {
              title: "Natural Language Processing",
              text: "Analyzes news and social media feeds to assign sentiment scores to market trends."
            },
            {
              title: "Reinforcement Learning",
              text: "Simulates trading strategies to identify the most profitable actions."
            },
          ].map((comp, index) => (
            <div key={index} style={{ backgroundColor: "#fff", padding: "1.5rem", borderRadius: "10px", boxShadow: "0 4px 8px rgba(0,0,0,0.1)" }}>
              <h3 style={{ fontSize: "1.5rem", marginBottom: ".5rem" }}>{comp.title}</h3>
              <p>{comp.text}</p>
            </div>
          ))}
        </div>
      </section>

      <footer style={{ marginTop: "4rem", padding: "2rem", backgroundColor: "#1c1c1e", color: "#fff", borderRadius: "8px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "2rem" }}>
          <div>
            <h3>Contact Us</h3>
            <p><a href="mailto:ajp12@bu.edu" style={{ color: "#80d0ff" }}>ajp12@bu.edu</a></p>
            <p><a href="mailto:gabrown@bu.edu" style={{ color: "#80d0ff" }}>gabrown@bu.edu</a></p>
            <p><a href="mailto:ykaan@bu.edu" style={{ color: "#80d0ff" }}>ykaan@bu.edu</a></p>
            <p><a href="mailto:odilon@bu.edu" style={{ color: "#80d0ff" }}>odilon@bu.edu</a></p>
          </div>
          <div>
            <h3>Useful Links</h3>
            <p><a href="/UserManual.pdf" target="_blank" style={{ color: "#80d0ff" }}>User Manual</a></p>
            <p><a href="/PrivacyPolicy.pdf" target="_blank" style={{ color: "#80d0ff" }}>Privacy Policy</a></p>
            <p><a href="/TermsAndConditions.pdf" target="_blank" style={{ color: "#80d0ff" }}>Terms and Conditions</a></p>
          </div>
          <div>
            <h3>Follow Us</h3>
            <p><a href="#" style={{ color: "#80d0ff" }}>YouTube</a></p>
            <p><a href="#" style={{ color: "#80d0ff" }}>Instagram</a></p>
            <p><a href="#" style={{ color: "#80d0ff" }}>WhatsApp</a></p>
            <p><a href="#" style={{ color: "#80d0ff" }}>GitHub</a></p>
          </div>
        </div>
        <div style={{ marginTop: "2rem", textAlign: "center" }}>
          Created by <span style={{ color: "coral" }}>EC464 Team 6</span> | All rights reserved.
        </div>
      </footer>
    </div>
  );
};



