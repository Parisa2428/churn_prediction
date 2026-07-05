"""
Pydantic schemas for FastAPI
Defines the structure of request and response data
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChurnRequest(BaseModel):
    """Schema for churn prediction request"""
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=85.7)
    TotalCharges: Optional[float] = Field(None, ge=0, example=1020.0)
    
    class Config:
        json_schema_extra = {
            "example": {
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
                "TotalCharges": 1020.0,
            }
        }


class ChurnResponse(BaseModel):
    """Schema for churn prediction response"""
    prediction: int = Field(..., description="0: No Churn, 1: Churn")
    prediction_label: str = Field(..., description="No Churn or Churn")
    probability: float = Field(..., description="Churn probability")
    risk_level: str = Field(..., description="Risk level")
    model_name: str = Field(..., description="Model name used")


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    model_name: str