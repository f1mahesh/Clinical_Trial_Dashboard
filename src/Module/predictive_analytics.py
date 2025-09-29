"""
Predictive Analytics Page
========================

Advanced predictive analytics and machine learning models.
"""

import streamlit as st
from src.utils.common_service import check_database_connection
import plotly.express as px
import numpy as np
import pandas as pd

def render_predictive_analytics_page():
    """Render the predictive analytics page"""
    st.header("🤖 Predictive Analytics & Advanced Insights",divider=True)
    
    # Check if database is connected
    if not check_database_connection():
        return
    
        
    st.markdown("""
    <div class="insight-box">
    <h5>🔮 Predictive Models</h5>
    <p>Advanced analytics using machine learning to predict trial outcomes and identify patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for advanced analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Trial Success Prediction Model")
        
        # Simulate a prediction model interface
        phase_input = st.selectbox("Study Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
        enrollment_input = st.slider("Enrollment Size", 10, 2000, 100)
        sponsor_input = st.selectbox("Sponsor Type", ["Industry", "Academic", "Government"])
        
        if st.button("🎯 Predict Success Probability"):
            # Simulate prediction (in real implementation, this would use trained ML model)
            success_prob = np.random.uniform(0.4, 0.9)
            st.metric("Predicted Success Probability", f"{success_prob:.1%}")
            
            if success_prob > 0.7:
                st.success("🟢 High probability of success")
            elif success_prob > 0.5:
                st.warning("🟡 Moderate probability of success")
            else:
                st.error("🔴 Lower probability of success")
    
    with col2:
        st.subheader("🎯 Enrollment Forecasting")
        
        # Simulate enrollment forecasting
        condition_input = st.selectbox("Medical Condition", 
                                        ["Cancer", "Diabetes", "Hypertension", "Depression"])
        
        if st.button("📈 Forecast Enrollment Timeline"):
            # Generate sample forecast data
            months = list(range(1, 25))
            enrollment_forecast = [
                int(enrollment_input * (1 - np.exp(-0.1 * m)) + np.random.normal(0, 5))
                for m in months
            ]
            
            fig_forecast = px.line(
                x=months,
                y=enrollment_forecast,
                title="Enrollment Forecast Over Time",
                labels={"x": "Months", "y": "Cumulative Enrollment"}
            )
            fig_forecast.add_hline(
                y=enrollment_input, 
                line_dash="dash", 
                annotation_text="Target Enrollment"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Risk assessment
    st.subheader("⚠️ Risk Assessment Dashboard")
    
    risk_factors = {
        "Protocol Complexity": np.random.uniform(0.2, 0.8),
        "Regulatory Risk": np.random.uniform(0.1, 0.6),
        "Competitive Landscape": np.random.uniform(0.3, 0.9),
        "Recruitment Feasibility": np.random.uniform(0.4, 0.8)
    }
    
    risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Risk Factor', 'Risk Level'])
    
    fig_risk = px.bar(
        risk_df,
        x='Risk Factor',
        y='Risk Level',
        title="Trial Risk Assessment",
        color='Risk Level',
        color_continuous_scale='RdYlGn_r'
    )
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

render_predictive_analytics_page()