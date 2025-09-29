"""
Innovation Insights Page
=======================

Analyzes innovation trends, intervention types, and market insights
for clinical trials.
"""

import streamlit as st
from src.utils.common_service import check_database_connection,get_innovation_metrics,create_visualizations




def render_innovation_insights_page():
    """Render the innovation insights page"""
    
    st.header("💡 Innovation & Market Insights",divider=True)
    
     # Check if database is connected
    if not check_database_connection():
        return

       
    metrics = get_innovation_metrics()
        
    # Key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if not metrics['intervention_trends'].empty:
            top_intervention = metrics['intervention_trends'].iloc[0]
            st.metric(
                "Leading Intervention Type",
                top_intervention['intervention_type'],
                f"{top_intervention['studies_count']} studies"
            )
    
    with col2:
        if not metrics['emerging_areas'].empty:
            top_area = metrics['emerging_areas'].iloc[0]
            st.metric(
                "Top Therapeutic Area",
                top_area['condition'],
                f"{top_area['studies_count']} studies"
            )
    
    # Visualizations
    figures = create_visualizations(metrics, 'innovation')
    
    for title, fig in figures:
        st.plotly_chart(fig, use_container_width=True)
    
    # Innovation insights
    st.markdown("""
    <div class="success-box">
    <h4>🚀 Innovation Highlights</h4>
    <ul>
    <li>Drug interventions continue to dominate clinical research</li>
    <li>Emerging areas show increased industry investment</li>
    <li>Novel therapeutic approaches are gaining traction</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    

render_innovation_insights_page()