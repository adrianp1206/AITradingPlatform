# AI Trading Platform

## Quick-start & Gotchas

The AI Trading Platform is a modular, end-to-end system for algorithmic stock trading research and deployment. It comprises two main components:

1. **ML Models & Backfill** (`ML/`)
2. **Java Web Application** (`SeniorDesignWebApp/`)

### 🚀 Current State

- **ML/**: data pipelines, model training/inference, backfill script (Firestore), scheduler.
- **SeniorDesignWebApp/**: monolithic Java backend (Gradle) + React front-end.

### 🔑 Gotchas & Tips

1. **Firebase credentials**: place `firebase.json` in the project root (one level above `ML/`).
2. **Alpha Vantage rate limits**: use caching or your own API key.
3. **Environments**:
   - **Python**: use `ML/environment.yml` (Conda) or `ML/requirements.txt` (pip).
   - **Java**: ensure JDK ≥11; use the provided Gradle wrapper (`./gradlew`).
4. **Trading days**: backfill script skips weekends but not market holidays.
5. **Model versions**: artifacts in `ML/models/` must match your training library versions.
6. **Testing**:
   - **ML**: no automated tests; consider adding pytest suites under `ML/`.
   - **Webapp**: run `SeniorDesignWebApp/serviceTest.sh` for integration tests.
7. **Deployment**: no CI/CD; recommend containerizing services + health checks.

## 📦 Directory Layout

```text
/AITradingPlatform
├── ML/
│   ├── data_processing.py
│   ├── run_all_model.py
│   ├── backfill_script.py
│   ├── scheduler.py
│   ├── requirements.txt
│   ├── environment.yml
│   ├── lstm/  boost/  nlp/  rl/
│   └── models/
└── SeniorDesignWebApp/
    ├── monolith-service/
    ├── webapp/
    ├── serviceTest.sh
    └── build scripts (Gradle)
```

---

*Add any additional notes or sections below as needed.*
