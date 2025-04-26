# README_SOFTWARE.md (Code Documentation)

## 1. Module Overviews

### ML/
- **data_processing.py**: fetch historic data (Alpha Vantage) + compute technical indicators.
- **run_all_model.py**: orchestrates:
  - `predict_lstm()` (Keras LSTM)
  - `predict_xgboost()` (XGBoost)
  - `predict_nlp()` (HuggingFace sentiment)
  - `predict_rl()` (DQN agent)
- **backfill_script.py**: loops business days → runs inference → writes to Firestore.
- **scheduler.py**: example scheduler (cron or Python).
- **lstm/, boost/, nlp/, rl/**: training notebooks + serialized model artifacts.

**Backtest Results**

All backtests are stored as Jupyter notebooks in `ML/`:

- `lstm_backtest_results.ipynb`
- `xgboost_backtest_results.ipynb`
- `nlp_backtest_results.ipynb`
- `rl_backtest_results.ipynb`

### SeniorDesignWebApp/
- **monolith-service/**: Java REST API:
  - `/subscribe`, `/unsubscribe`
  - `/price/{symbol}`
  - `/mostactive`
- **webapp/**: React (Vite) front-end consuming backend APIs.
- **serviceTest.sh**: integration tests with mock FinHub client.

## 2. Dependency Flow

```text
 data_processing.py
        ↓
 run_all_model.py ←─ lstm/, boost/, nlp/, rl/
        ↓
 backfill_script.py → Firestore → scheduler.py

 SeniorDesignWebApp:
  webapp (frontend)
      ↑
      | HTTP
      ↓
  monolith-service (Java)
```

## 3. Dev / Build Tools

- **Python** ≥3.8 (3.9.16 recommended)
- **Conda**: `conda env create -f ML/environment.yml`
- **Pip**: `pip install -r ML/requirements.txt`
- **Java**: JDK ≥11
- **Gradle**: `./gradlew`
- **Node.js** ≥16

## 4. Installation & Build

1. **Clone**:
   ```bash
   git clone https://github.com/adrianp1206/AITradingPlatform.git
   cd AITradingPlatform
   ```
2. **Python env**:
   ```bash
   cd ML
   conda env create -f environment.yml
   # or
   pip install -r requirements.txt
   ```  
3. **Credentials**:
   - Place `firebase.json` in repo root.
   - Set `ALPHA_VANTAGE_KEY` env var or update `data_processing.py`.
4. **Java backend**:
   ```bash
   cd ../SeniorDesignWebApp
   ./gradlew clean build
   ./serviceTest.sh
   ```  
5. **Run**:
   - **ML backfill**: `cd ML && python backfill_script.py`
   - **Java service**: `cd SeniorDesignWebApp && ./gradlew runServer`
   - **Front-end**: `cd SeniorDesignWebApp/webapp && npm install && npm run dev`

---

*End of README_SOFTWARE.md.*
