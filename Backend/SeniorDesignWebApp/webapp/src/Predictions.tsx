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
  const [search, setSearch] = React.useState("");

  const toggleGraph = (ticker: string) => {
    setExpandedTicker(prev => (prev === ticker ? null : ticker));
  };

  const filteredPredictions = Object.values(
    predictions
      .filter(p => p.ticker.toLowerCase().includes(search.toLowerCase()))
      .reduce((acc, curr) => {
        if (!acc[curr.ticker] || new Date(curr.updated_at) > new Date(acc[curr.ticker].updated_at)) {
          acc[curr.ticker] = curr;
        }
        return acc;
      }, {} as Record<string, PredictionData>)
  );

  const getRLPerformanceForTicker = (ticker: string): RlPerformanceEntry[] =>
    rlPerformance
      .filter(e => e.ticker.toLowerCase() === ticker.toLowerCase())
      .sort((a, b) => a.date.localeCompare(b.date));

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
    const totalGain = data[data.length - 1].cumulative_pl / 100 + 1;
    const annualized = (Math.pow(totalGain, 365 / daysHeld) - 1) * 100;
    return isFinite(annualized) ? annualized : 0;
  };

  const getPillStyle = (value: string | number) => {
    const positive =
      value === "Buy" ||
      value === "Up" ||
      (typeof value === "number" && value > 0.5);
    return {
      backgroundColor: positive ? "#d1e7dd" : "#f8d7da",
      color: positive ? "#0f5132" : "#842029",
      borderRadius: "12px",
      padding: "4px 10px",
      fontWeight: 600,
      display: "inline-block",
      minWidth: "60px",
      textAlign: "center" as const,
    };
  };

  const containerStyle: React.CSSProperties = {
    padding: "2rem",
    fontFamily: "Montserrat, sans-serif",
    background: "#f0f4f8",
    borderRadius: "12px",
    color: "#000",
    width: "90vw",        
    overflowX: "hidden",    
    boxSizing: "border-box",
  };
  

  const cardStyle: React.CSSProperties = {
    background: "#fff",
    borderRadius: "8px",
    padding: "1rem 2rem",
    margin: "1.5rem 0",
    boxShadow: "0 4px 6px rgba(0,0,0,0.1)",
    display: "flex",
    flexDirection: "column",
    width: "100%",
    boxSizing: "border-box",
    gap: "1rem",
  };

  const headerRowStyle: React.CSSProperties = {
    display: "flex",
    width: "100%",
    alignItems: "center",
    justifyContent: "space-between", 
    flexWrap: "wrap" as const,
    gap: "1rem",
  };

  const blockStyle: React.CSSProperties = {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    flex: "1",             
    minWidth: "120px",
  };

  const dividerStyle: React.CSSProperties = {
    height: "60px",
    width: "1px",
    backgroundColor: "#ccc",
  };

  // =============================================================================
  // RENDER
  // =============================================================================
  return (
    <div style={containerStyle}>
      <h2 style={{ textAlign: "center", fontSize: "2rem", marginBottom: "2rem" }}>
        Today's Recommendations
      </h2>

      <input
        type="text"
        placeholder="Search by ticker..."
        value={search}
        onChange={e => setSearch(e.target.value)}
        style={{
          padding: "0.5rem",
          fontSize: "1rem",
          margin: "0 auto 1.5rem",
          width: "100%",
          maxWidth: "300px",
          borderRadius: "8px",
          border: "1px solid #ccc",
          display: "block",
        }}
      />

      {filteredPredictions.map(p => {
        const totalGain = getTotalGain(p.ticker);
        const yoyGain = getYOYGain(p.ticker);
        const isExpanded = expandedTicker === p.ticker;
        const perfData = getRLPerformanceForTicker(p.ticker);

        return (
          <div key={p.ticker} style={cardStyle}>
            <div style={headerRowStyle}>
              {/* Ticker & Toggle */}
              <div style={blockStyle}>
                <span style={{ fontSize: "1.5rem", fontWeight: 700 }}>{p.ticker}</span>
                <span
                  onClick={() => toggleGraph(p.ticker)}
                  style={{
                    cursor: "pointer",
                    marginTop: "4px",
                    fontSize: "1rem",
                    fontWeight: 600,
                    color: totalGain < 0 ? "#dc3545" : "#198754",
                  }}
                >
                  {isExpanded ? "▼ Hide P/L" : "▶ Show P/L"} • Total:{" "}
                  {totalGain.toFixed(2)}%
                </span>
                <span style={{ fontSize: "0.75rem", color: "#444" }}>
                  Est. YOY: {yoyGain.toFixed(2)}%
                </span>
              </div>

              <div style={dividerStyle} />

              {/* LSTM */}
              <div style={blockStyle}>
                <strong>LSTM Price</strong>
                <span>${p.lstm_predicted_price.toFixed(2)}</span>
              </div>

              <div style={dividerStyle} />

              {/* XGBoost */}
              <div style={blockStyle}>
                <strong>XGBoost</strong>
                <span style={getPillStyle(p.xgboost_signal)}>
                  {p.xgboost_signal}
                </span>
                <div style={{ fontSize: "0.75rem", marginTop: "4px" }}>
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
                  />
                </div>
              </div>

              <div style={dividerStyle} />

              {/* NLP */}
              <div style={blockStyle}>
                <strong>NLP Score</strong>
                <span style={getPillStyle(p.nlp_sentiment_score)}>
                  {p.nlp_sentiment_score.toFixed(2)}
                </span>
              </div>

              <div style={dividerStyle} />

              {/* RL Basic */}
              <div style={blockStyle}>
                <strong>RL</strong>
                <span style={getPillStyle(p.rl_recommendation)}>
                  {p.rl_recommendation}
                </span>
              </div>

              <div style={dividerStyle} />

              {/* Last Updated */}
              <div style={blockStyle}>
                <em style={{ fontSize: "12px", color: "#777" }}>Last Updated</em>
                <span style={{ fontSize: "12px", color: "#777" }}>
                  {new Date(p.updated_at).toLocaleString()}
                </span>
              </div>
            </div>

            {/* P/L Chart */}
            {isExpanded && (
              <div
                style={{
                  width: "100%",
                  marginTop: "1rem",
                  backgroundColor: "#f9f9f9",
                  padding: "1rem",
                  borderRadius: "8px",
                  boxShadow: "inset 0 0 3px rgba(0,0,0,0.1)",
                }}
              >
                {perfData.length === 0 ? (
                  <div style={{ textAlign: "center", color: "#888" }}>
                    No P/L data available
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={perfData}>
                      <XAxis dataKey="date" />
                      <YAxis domain={["auto", "auto"]} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="cumulative_pl"
                        stroke={totalGain < 0 ? "#dc3545" : "#198754"}
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
