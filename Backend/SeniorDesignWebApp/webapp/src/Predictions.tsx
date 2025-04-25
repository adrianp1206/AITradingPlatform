import * as React from "react";
import { useLoaderData } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
interface PredictionData {
  ticker: string;
  lstm_predicted_price: number;
  nlp_sentiment_score: number;
  rl_recommendation: string;
  updated_at: string;
  xgboost_signal: string;
  xgboost_prob: number;
  prediction_id: string;
  next_trading_date: string;
}
interface RlPerformanceEntry {
  ticker: string;
  date: string;
  cumulative_pl: number;
}

export const Predictions: React.FC = () => {
  const { predictions, rlPerformance } = useLoaderData() as {
    predictions: PredictionData[];
    rlPerformance: RlPerformanceEntry[];
  };
  const [expandedTicker, setExpandedTicker] = React.useState<string | null>(null);

  const toggleGraph = (ticker: string) => {
    setExpandedTicker(prev => (prev === ticker ? null : ticker));
  };
  const [search, setSearch] = React.useState("");
  const filteredPredictions = Object.values(
    predictions
      .filter((p) =>
        p.ticker.toLowerCase().includes(search.toLowerCase())
      )
      .reduce((acc, curr) => {
        if (!acc[curr.ticker] || new Date(curr.updated_at) > new Date(acc[curr.ticker].updated_at)) {
          acc[curr.ticker] = curr;
        }
        return acc;
      }, {} as { [ticker: string]: PredictionData })
  );
  const getRLPerformanceForTicker = (ticker: string): RlPerformanceEntry[] => {
    return rlPerformance
    .filter((entry) => entry.ticker.toLowerCase() === ticker.toLowerCase())
    .sort((a, b) => a.date.localeCompare(b.date));
  };
  const getTotalGain = (ticker: string): number => {
    const data = getRLPerformanceForTicker(ticker);
    if (data.length === 0) return 0;
    return data[data.length - 1].cumulative_pl;
  };
  
  const getYOYGain = (ticker: string): number => {
    const data = getRLPerformanceForTicker(ticker);
    if (data.length < 2) return 0;
  
    const startDate = new Date(data[0].date);
    const endDate = new Date(data[data.length - 1].date);
    const daysHeld = (endDate.getTime() - startDate.getTime()) / (1000 * 3600 * 24);
    const totalGain = data[data.length - 1].cumulative_pl;
  
    const annualizedReturn = ((1 + totalGain / 100) ** (365 / daysHeld) - 1) * 100;
    return isFinite(annualizedReturn) ? annualizedReturn : 0;
  };
  const getPill = (value: string | number) => {
    const isPositive =
      value === "Buy" || value === "Up" || (typeof value === "number" && value > 0.5);
    return {
      backgroundColor: isPositive ? "#d1e7dd" : "#f8d7da",
      color: isPositive ? "#0f5132" : "#842029",
      borderRadius: "12px",
      padding: "4px 10px",
      fontWeight: 600,
      display: "inline-block",
      minWidth: "60px"
    };
  };

  const containerStyle: React.CSSProperties = {
    padding: "2rem",
    fontFamily: "Montserrat, sans-serif",
    background: "#f0f4f8",
    borderRadius: "12px",
    color: "#000",
    maxWidth: "100%",
    margin: "0 auto",
  };

  const cardStyle: React.CSSProperties = {
    background: "#ffffff",
    borderRadius: "8px",
    padding: "1rem 2rem",
    margin: "1.5rem 0",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    overflowX: "auto",
    gap: "1.5rem",
    width: "100%",
    boxSizing: "border-box",
  };

  const blockStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    minWidth: "140px",
  };

  const dividerStyle: React.CSSProperties = {
    height: "60px",
    width: "1px",
    backgroundColor: "#ccc",
  };

  return (
    <div style={containerStyle}>
      <h2 style={{ textAlign: "center", fontSize: "2rem", marginBottom: "2rem" }}>
        Today's Recommendations
      </h2>

      <input
        type="text"
        placeholder="Search by ticker..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{
          padding: "0.5rem",
          fontSize: "1rem",
          marginBottom: "1.5rem",
          width: "100%",
          maxWidth: "300px",
          borderRadius: "8px",
          border: "1px solid #ccc",
          display: "block",
          marginLeft: "auto",
          marginRight: "auto",
        }}
      />

{filteredPredictions.map((p) => {
  const isExpanded = expandedTicker?.toLowerCase() === p.ticker.toLowerCase();
  const perfData = getRLPerformanceForTicker(p.ticker);

  return (
    <div key={p.ticker} style={{ ...cardStyle, flexDirection: "column" }}>
      <div style={{ display: "flex", width: "100%", justifyContent: "space-between", alignItems: "center" }}>
        <div style={blockStyle}>
          <span style={{ fontSize: "1.5rem", fontWeight: 700 }}>{p.ticker}</span>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              marginTop: "4px",
              cursor: "pointer",
            }}
            onClick={() => toggleGraph(p.ticker)}
          >
            <span style={{ fontSize: "1.2rem", color: "#555" }}>
              {isExpanded ? "⬆ Hide P/L" : "⬇ Show P/L"}
            </span>
            <span style={{ fontSize: "0.85rem", color: "#00c853", fontWeight: 600 }}>
              Total: {getTotalGain(p.ticker).toFixed(2)}%
            </span>
            <span style={{ fontSize: "0.75rem", color: "#444" }}>
              Est. YOY: {getYOYGain(p.ticker).toFixed(2)}%
            </span>
          </div>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <strong>LSTM Price</strong>
          <span>${p.lstm_predicted_price.toFixed(2)}</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <strong>XGBoost</strong>
          <span style={getPill(p.xgboost_signal)}>{p.xgboost_signal}</span>
          <div style={{ fontSize: "0.75rem", marginTop: "6px", color: "#444" }}>
            Confidence
          </div>
          <div
            style={{
              height: "6px",
              background: "#ddd",
              borderRadius: "3px",
              marginTop: "4px",
              width: "80px",
            }}
          >
            <div
              style={{
                width: `${(p.xgboost_prob * 100).toFixed(1)}%`,
                background: "#0d6efd",
                height: "100%",
                borderRadius: "3px",
              }}
            ></div>
          </div>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <strong>NLP Score</strong>
          <span style={getPill(p.nlp_sentiment_score)}>
            {p.nlp_sentiment_score.toFixed(2)}
          </span>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <strong>RL (Basic)</strong>
          <span style={getPill(p.rl_recommendation)}>{p.rl_recommendation}</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <strong>RL (Shortable)</strong>
          <span style={{ color: "#aaa" }}>N/A</span>
        </div>

        <div style={dividerStyle}></div>

        <div style={blockStyle}>
          <em style={{ fontSize: "12px", color: "#777" }}>Last Updated</em>
          <span style={{ fontSize: "12px", color: "#777" }}>
            {new Date(p.updated_at).toLocaleString()}
          </span>
        </div>
      </div>

      {isExpanded && (
        <div style={{
          width: "100%",
          marginTop: "20px",
          backgroundColor: "#f9f9f9",
          padding: "1rem",
          borderRadius: "8px",
          boxShadow: "inset 0 0 3px rgba(0,0,0,0.1)",
        }}>
          {perfData.length === 0 ? (
            <div style={{ textAlign: "center", color: "#888" }}>
              No P/L data available
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={perfData}>
                <XAxis dataKey="date" />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="cumulative_pl"
                  stroke="#00c853"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      )}
    </div>
  );
})}

    </div>
  );
};
