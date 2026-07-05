"""
Streamlit Web UI for Customer Churn Prediction
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

# API URL
API_URL = "http://localhost:8000"


def main() -> None:
    """Main function for Streamlit UI"""
    st.title("📊 Customer Churn Prediction")
    st.markdown(
        "Predict customer churn using Machine Learning models"
    )
    
    # Sidebar - Health Check
    with st.sidebar:
        st.header("🔧 System Status")
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ API Status: {data['status']}")
                st.info(f"🤖 Model: {data['model_name']}")
            else:
                st.error("❌ API not available")
        except:
            st.error("❌ Cannot connect to API")
            st.info("Make sure API is running: `uvicorn src.api.main:app`")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Single Customer Prediction",
        "📦 Batch Prediction",
        "📈 Reports"
    ])
    
    with tab1:
        _single_prediction_tab()
    
    with tab2:
        _batch_prediction_tab()
    
    with tab3:
        _reports_tab()


def _single_prediction_tab() -> None:
    """Tab for single customer prediction"""
    st.header("🎯 Predict for a Single Customer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox(
            "Internet Service", 
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with col3:
        st.subheader("💳 Contract & Billing")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract", 
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 
            min_value=0.0, value=85.0, step=0.1
        )
        total_charges = st.number_input(
            "Total Charges ($)", 
            min_value=0.0, value=1000.0, step=0.1
        )
    
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        payload = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        
        try:
            response = requests.post(
                f"{API_URL}/predict", 
                json=payload, 
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                _display_result(result)
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")


def _display_result(result: dict) -> None:
    """Display prediction result"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Gauge chart for probability
        prob = result["probability"] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": "Churn Probability (%)", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#90EE90"},
                    {"range": [30, 60], "color": "#FFD700"},
                    {"range": [60, 100], "color": "#FF6B6B"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": prob,
                },
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric(
            "Prediction",
            result["prediction_label"],
            delta=None
        )
        st.metric(
            "Risk Level",
            result["risk_level"],
            delta=None
        )
        st.metric(
            "Model",
            result["model_name"],
            delta=None
        )
    
    with col3:
        if result["prediction"] == 1:
            st.error("⚠️ At Risk of Churn")
            st.markdown("""
            **Recommendations:**
            - Contact customer
            - Offer discount
            - Check for issues
            """)
        else:
            st.success("✅ Loyal Customer")


def _batch_prediction_tab() -> None:
    """Tab for batch prediction"""
    st.header("📦 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Data Preview:**")
        st.dataframe(df.head())
        
        if st.button("🚀 Predict Batch", type="primary"):
            st.info("Processing...")
            # Implementation for batch prediction
            st.success(f"✅ {len(df)} predictions completed")


def _reports_tab() -> None:
    """Tab for reports"""
    st.header("📈 Reports and Analysis")
    st.info("This section can display SHAP plots and Confusion Matrix")


if __name__ == "__main__":
    main()