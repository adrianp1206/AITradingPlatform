README.md (Engineering Addendum)

AI Trading Platform

Quick-start & Gotchas

The AI Trading Platform is a modular, end-to-end system for algorithmic stock trading research and deployment. It comprises two main components:

ML Models & Backfill (ML/)

Java Web Application (SeniorDesignWebApp/)

🚀 Current State

ML/: data pipelines, model training/inference, backfill script (Firestore), scheduler.

SeniorDesignWebApp/: monolithic Java backend (Gradle) + React front-end.

🔑 Gotchas & Tips

Firebase credentials: place firebase.json in the project root (one level above ML/).

Alpha Vantage rate limits: use caching or your own API key.

Environments:

Python: use ML/environment.yml (Conda) or ML/requirements.txt (pip).

Java: ensure JDK ≥11; use the provided Gradle wrapper (./gradlew).

Trading days: backfill script skips weekends but not market holidays.

Model versions: artifacts in ML/models/ must match your training library versions.

Testing:

ML: no automated tests; consider adding pytest suites under ML/.

Webapp: run SeniorDesignWebApp/serviceTest.sh for integration tests.

Deployment: no CI/CD; recommend containerizing services + health checks.

📦 Directory Layout

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


