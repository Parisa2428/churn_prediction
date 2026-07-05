"""
FastAPI REST API for Customer Churn Prediction
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import config
from src.api.schemas import ChurnRequest, ChurnResponse, HealthResponse


# Load model and preprocessor
logger.info("[API] Loading models...")
try:
    model = joblib.load(config.MODELS_DIR / "XGBoost.pkl")
    preprocessor = joblib.load(config.MODELS_DIR / "preprocessor.pkl")
    MODEL_NAME = "XGBoost"
    MODEL_LOADED = True
    logger.success(f"[API] Model {MODEL_NAME} loaded successfully")
except Exception as e:
    logger.error(f"[API] Error loading model: {e}")
    model = None
    preprocessor = None
    MODEL_NAME = "None"
    MODEL_LOADED = False


# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_risk_level(probability: float) -> str:
    """Determine risk level based on probability"""
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_LOADED,
        model_name=MODEL_NAME,
    )


@app.post("/predict", response_model=ChurnResponse)
async def predict_churn(request: ChurnRequest) -> ChurnResponse:
    """Predict customer churn"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data = request.model_dump()
        df = pd.DataFrame([data])
        
        # Preprocess
        X_processed = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        return ChurnResponse(
            prediction=int(prediction),
            prediction_label="Churn" if prediction == 1 else "No Churn",
            probability=float(probability),
            risk_level=_get_risk_level(probability),
            model_name=MODEL_NAME,
        )
    
    except Exception as e:
        logger.error(f"[API] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: list[ChurnRequest]) -> list[ChurnResponse]:
    """Batch prediction"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        data = [req.model_dump() for req in requests]
        df = pd.DataFrame(data)
        
        X_processed = preprocessor.transform(df)
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(ChurnResponse(
                prediction=int(pred),
                prediction_label="Churn" if pred == 1 else "No Churn",
                probability=float(prob),
                risk_level=_get_risk_level(float(prob)),
                model_name=MODEL_NAME,
            ))
        
        return results
    
    except Exception as e:
        logger.error(f"[API] Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)