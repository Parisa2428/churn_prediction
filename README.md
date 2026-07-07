# 📊 Customer Churn Prediction

A production-grade Machine Learning project for predicting customer churn in the telecommunications industry. Built with best practices from data preprocessing to deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

This project predicts whether a telecom customer will churn (leave the service) based on their demographic, service, and contract information. It implements a complete ML pipeline including:

- ✅ **Data preprocessing** with missing value handling, encoding, and scaling
- ✅ **Class imbalance handling** using SMOTE
- ✅ **Multiple ML models** with hyperparameter tuning via Optuna
- ✅ **Model explainability** using SHAP
- ✅ **REST API** with FastAPI
- ✅ **Web UI** with Streamlit
- ✅ **Containerized deployment** with Docker

---

## 🏗️ System Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Streamlit UI  │─────▶│   FastAPI API    │─────▶│  ML Model (.pkl)│
│   (Port 8501)   │      │   (Port 8000)    │      │   XGBoost       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Preprocessor   │
                         │  (ColumnTrans.)  │
                         └──────────────────┘
```

---

## 📁 Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                  # Raw dataset (Telco Customer Churn)
│   └── processed/            # Processed data
├── notebooks/                # EDA and experimentation
├── src/
│   ├── config.py             # Global configuration
│   ├── data/
│   │   ├── loader.py         # Data loading
│   │   └── preprocessing.py  # Data preprocessing
│   ├── features/             # Feature engineering
│   ├── models/
│   │   ├── trainer.py        # Model training
│   │   └── evaluator.py      # Model evaluation
│   ├── explainability/
│   │   └── shap_explainer.py # SHAP analysis
│   ├── api/
│   │   ├── main.py           # FastAPI endpoints
│   │   └── schemas.py        # Pydantic schemas
│   └── ui/
│       └── app.py            # Streamlit interface
├── models/                   # Trained models (pickle files)
├── reports/figures/          # Generated plots
├── tests/                    # Unit tests
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Multi-service orchestration
├── Makefile                  # Common commands
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Docker & Docker Compose (optional)

### Local Setup

#### 1. Clone the repository

```bash
git clone https://github.com/Parisa2428/churn_prediction.git
cd churn_prediction
```

#### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Train models

```bash
python main.py
```

This will:
- Download the dataset (if not present)
- Preprocess the data
- Train 3 models (Logistic Regression, Random Forest, XGBoost)
- Evaluate models and generate plots
- Perform SHAP analysis

#### 5. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Access API docs at: http://localhost:8000/docs

#### 6. Start the UI (in another terminal)

```bash
streamlit run src/ui/app.py
```

Access UI at: http://localhost:8501

---

### Docker Setup

```bash
# Build and run all services
docker-compose up --build

# Or use Makefile
make docker-up
```

Access:
- **API**: http://localhost:8000/docs
- **UI**: http://localhost:8501

---

## 📊 Dataset

**Telco Customer Churn** - 7,043 customers with 21 features

- **Source**: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Samples**: 7,043
- **Features**: 21 (demographics, services, contract)
- **Target**: Churn (Yes/No)
- **Class Distribution**: 73.46% No Churn, 26.54% Churn (imbalanced)

---

## 🤖 Models

Three models were trained with hyperparameter tuning using Optuna:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.7374 | 0.5034 | 0.7941 | 0.6162 | **0.8393** |
| Random Forest | 0.7736 | 0.5711 | 0.5909 | 0.5808 | 0.8218 |
| XGBoost | 0.7786 | 0.5824 | 0.5856 | 0.5840 | 0.8260 |

**Selected Model**: Logistic Regression (best ROC-AUC)

### Key Techniques

- **SMOTE**: Applied only on training data to avoid data leakage
- **Optuna**: Bayesian optimization for hyperparameter tuning
- **Stratified K-Fold**: 5-fold cross-validation
- **Class Balancing**: `class_weight="balanced"` for linear models

---

## 🔍 Explainability (SHAP)

Top features driving churn:

1. `Contract_Month-to-month` - Month-to-month contracts have highest churn risk
2. `tenure` - Longer tenure reduces churn probability
3. `MonthlyCharges` - Higher charges increase churn risk
4. `InternetService_Fiber optic` - Fiber optic users churn more
5. `PaymentMethod_Electronic check` - Electronic check payers churn more

---

## 📈 Results

### Confusion Matrix
![Confusion Matrix](reports/figures/confusion_matrices.png)

### ROC Curves
![ROC Curves](reports/figures/roc_curves.png)

### Metrics Comparison
![Metrics Comparison](reports/figures/metrics_comparison.png)

### SHAP Summary
![SHAP Summary](reports/figures/shap_summary_LogisticRegression.png)

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch prediction |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.7,
    "TotalCharges": 1020.0
  }'
```

### Example Response

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "probability": 0.78,
  "risk_level": "High",
  "model_name": "XGBoost"
}
```

---

## 🛠️ Makefile Commands

```bash
make help          # Show all available commands
make install       # Install dependencies
make train         # Train models
make api           # Run FastAPI
make ui            # Run Streamlit
make docker-up     # Start Docker services
make docker-down   # Stop Docker services
make clean         # Clean temporary files
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📝 Key Decisions

| Decision | Rationale |
|----------|-----------|
| **SMOTE only on train** | Prevents data leakage |
| **Optuna over GridSearch** | Faster, smarter search with Bayesian optimization |
| **ROC-AUC as main metric** | Robust for imbalanced datasets |
| **SHAP over LIME** | Stronger theoretical foundation, global + local explanations |
| **FastAPI** | Async, auto-generated docs, high performance |
| **Docker multi-stage build** | Smaller final image, faster deployment |

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Built as a BSc thesis project demonstrating production-grade ML engineering practices.

**GitHub**: [Parisa2428](https://github.com/Parisa2428)

---

## 🙏 Acknowledgments

- Dataset: IBM Telco Customer Churn
- Libraries: scikit-learn, XGBoost, Optuna, SHAP, FastAPI, Streamlit