import * as React from "react";
import { Link, Outlet, useNavigate } from "react-router-dom";

export const StockApp: React.FC = () => {
  const navigate = useNavigate();

  const goHome = React.useCallback(() => {
    navigate("/");
  }, [navigate]);

  return (
    <>
      <header
        style={{
          backgroundColor: "#1c1c1e",
          padding: "1rem 2rem",
          borderRadius: "8px",
          marginBottom: "2rem",
          textAlign: "center",
        }}
      >
        <h1
          onClick={goHome}
          style={{
            color: "#fff",
            fontSize: "2.5rem",
            fontWeight: "bold",
            margin: 0,
            cursor: "pointer",
          }}
        >
          <i
            className="fas fa-money-bill-trend-up"
            style={{ marginRight: ".5rem", color: "coral" }}
          ></i>
          AI Trading Platform
        </h1>
        <nav style={{ marginTop: "1rem" }}>
          <Link
            to="/"
            style={{ color: "#80d0ff", margin: "0 1rem", fontSize: "1.2rem" }}
          >
            Home
          </Link>
          <Link
            to="/about"
            style={{ color: "#80d0ff", margin: "0 1rem", fontSize: "1.2rem" }}
          >
            About
          </Link>
          <Link
            to="/predictions"
            style={{ color: "#80d0ff", margin: "0 1rem", fontSize: "1.2rem" }}
          >
            Predictions
          </Link>
        </nav>
      </header>

      {/* Render sub-routes here */}
      <main style={{ padding: "0 2rem" }}>
        <Outlet />
      </main>
    </>
  );
};
