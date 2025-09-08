import React, { useState } from "react";
import axios from "axios";
import type { PredictionResponse } from "./types";

interface StockOption {
  symbol: string;
  name: string;
  exchange: string;
}

const App: React.FC = () => {
  // Styles
  const styles = {
    container: {
      padding: "20px",
      fontFamily: "Arial, sans-serif",
      maxWidth: "800px",
      margin: "0 auto",
    },
    spinner: {
      width: "40px",
      height: "40px",
      margin: "20px auto",
      border: "4px solid #f3f3f3",
      borderTop: "4px solid #3498db",
      borderRadius: "50%",
      animation: "spin 1s linear infinite",
    },
    "@keyframes spin": {
      "0%": { transform: "rotate(0deg)" },
      "100%": { transform: "rotate(360deg)" },
    },
    title: {
      color: "#2c3e50",
      marginBottom: "30px",
      textAlign: "center",
    },
    searchContainer: {
      position: "relative",
      width: "300px",
      margin: "0 auto",
    },
    input: {
      width: "100%",
      padding: "10px 15px",
      fontSize: "16px",
      border: "2px solid #ddd",
      borderRadius: "4px",
      outline: "none",
      boxSizing: "border-box",
      transition: "border-color 0.3s ease",
    },
    dropdown: {
      listStyle: "none",
      padding: 0,
      margin: "5px 0",
      position: "absolute",
      top: "100%",
      left: 0,
      right: 0,
      backgroundColor: "white",
      border: "1px solid #ddd",
      borderRadius: "4px",
      boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
      zIndex: 1000,
      maxHeight: "300px",
      overflowY: "auto",
    },
    dropdownItem: {
      padding: "10px 15px",
      cursor: "pointer",
      borderBottom: "1px solid #eee",
      "&:hover": {
        backgroundColor: "#f5f5f5",
      },
    },
    symbol: {
      fontWeight: "bold",
    },
    companyName: {
      fontSize: "0.9em",
      color: "#666",
    },
    exchange: {
      fontSize: "0.8em",
      color: "#999",
    },
  };
  const [query, setQuery] = useState<string>("");
  const [options, setOptions] = useState<StockOption[]>([]);
  const [ticker, setTicker] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // 🔍 Search company names
  const handleSearch = async (value: string) => {
    setQuery(value);
    if (value.length < 2) {
      setOptions([]);
      return;
    }
    const res = await axios.get<StockOption[]>(
      `http://127.0.0.1:8000/search?query=${value}`
    );
    setOptions(res.data);
  };

  // 📈 Predict
  const handlePredict = async () => {
    if (!ticker) return;
    setLoading(true);
    setError(null);

    try {
      const res = await axios.get<PredictionResponse>(
        `http://127.0.0.1:8000/predict?ticker=${ticker}`
      );
      if (res.data.error) {
        setError(res.data.error);
      } else {
        setPrediction(res.data);
      }
    } catch (err) {
      setError("Failed to fetch prediction. Is the API running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        padding: "20px",
        fontFamily: "Arial, sans-serif",
        maxWidth: "800px",
        margin: "0 auto",
      }}
    >
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
          .spinner {
            width: 40px;
            height: 40px;
            margin: 20px auto;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
          }
        `}
      </style>
      <h1
        style={{ color: "#2c3e50", marginBottom: "30px", textAlign: "center" }}
      >
        📈 Stock Price Predictor
      </h1>

      <div style={{ position: "relative", width: "300px", margin: "0 auto" }}>
        <input
          type="text"
          placeholder="Search company (e.g., Infosys, Apple)"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          style={{
            width: "100%",
            padding: "10px 15px",
            fontSize: "16px",
            border: "2px solid #ddd",
            borderRadius: "4px",
            outline: "none",
            boxSizing: "border-box",
            transition: "border-color 0.3s ease",
          }}
        />

        {/* Autocomplete dropdown */}
        {options.length > 0 && (
          <ul
            style={{
              listStyle: "none",
              padding: 0,
              margin: "5px 0",
              position: "absolute",
              top: "100%",
              left: 0,
              right: 0,
              backgroundColor: "white",
              border: "1px solid #ddd",
              borderRadius: "4px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
              zIndex: 1000,
              maxHeight: "300px",
              overflowY: "auto",
            }}
          >
            {options.map((opt) => (
              <li
                key={opt.symbol}
                onClick={() => {
                  setTicker(opt.symbol);
                  setQuery(opt.name); // Keep the company name in the search box
                  setOptions([]);
                  // Trigger prediction immediately when an option is selected
                  setTimeout(() => handlePredict(), 0);
                }}
                style={{
                  padding: "10px 15px",
                  cursor: "pointer",
                  borderBottom: "1px solid #eee",
                  transition: "background-color 0.2s ease",
                  backgroundColor: "white",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "#f5f5f5";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "white";
                }}
              >
                <div style={{ fontWeight: "bold" }}>{opt.symbol}</div>
                <div style={{ fontSize: "0.9em", color: "#666" }}>
                  {opt.name}
                </div>
                <div style={{ fontSize: "0.8em", color: "#999" }}>
                  {opt.exchange}
                </div>
              </li>
            ))}
          </ul>
        )}

        <button onClick={handlePredict} style={{ padding: "5px 10px" }}>
          Predict
        </button>

        {loading && <div className="spinner" />}
        {error && <p style={{ color: "red" }}>{error}</p>}

        {prediction && !error && (
          <div style={{ marginTop: "20px" }}>
            <h2>{prediction.ticker}</h2>
            <p>
              <strong>Last Actual Price:</strong> {prediction.last_actual_price}
            </p>
            <p>
              <strong>Next Predicted Price:</strong>{" "}
              {prediction.next_predicted_price}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
