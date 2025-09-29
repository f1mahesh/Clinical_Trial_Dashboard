"""
Executive Summary Page
=====================

High-level executive summary and strategic insights.
"""

import streamlit as st
from src.utils.common_service import get_trial_landscape_metrics, get_success_completion_metrics, get_innovation_metrics, get_access_equity_metrics,check_database_connection 
from textwrap import dedent
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

CACHE_TTL=60*60*24

def render_executive_summary_page():
    """Render the executive summary page"""
    
    st.header("📈 Executive Summary",divider=True)
    
     # Check if database is connected
    if not check_database_connection():
        return
        
    
        
    # Get all metrics for summary
   
    # Use parallel processing to load metrics concurrently
    
    # Load metrics sequentially since ThreadPoolExecutor causes issues with Streamlit session state
    # The session state (st.session_state) is thread-local and not accessible from worker threads
    # This means db_manager and other session variables cannot be accessed in parallel executions
    landscape_metrics = get_trial_landscape_metrics()
    success_metrics = get_success_completion_metrics()
    innovation_metrics = get_innovation_metrics()
    access_metrics = get_access_equity_metrics()
    
    # landscape_metrics = get_trial_landscape_metrics()
    # success_metrics = get_success_completion_metrics()
    # innovation_metrics = get_innovation_metrics()
    # access_metrics = get_access_equity_metrics()

    # Executive dashboard
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Studies", f"{landscape_metrics['total_studies']:,}")
    
    with col2:
        if not success_metrics['completion_by_phase'].empty:
            avg_completion = success_metrics['completion_by_phase']['completion_rate'].mean()
            st.metric("Avg Completion Rate", f"{avg_completion:.1f}%")
    
    with col3:
        countries_count = len(access_metrics['geographic_access'])
        st.metric("Global Reach", f"{countries_count} countries")
    
    with col4:
        if not innovation_metrics['intervention_trends'].empty:
            total_interventions = innovation_metrics['intervention_trends']['studies_count'].sum()
            st.metric("Active Interventions", f"{total_interventions:,}")
    
    with col5:
        if not success_metrics['results_reporting'].empty:
            avg_reporting = success_metrics['results_reporting']['reporting_rate'].mean()
            st.metric("Results Reporting", f"{avg_reporting:.1f}%")
    
    # Summary insights
    st.subheader("📋 Strategic Insights")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Opportunities", "⚠️ Challenges", "🚀 Recommendations"])
    
    with tab1:
        html_content = dedent(f"""
        <div class="success-box">
            <h4>🎯 Market Opportunities</h4>
            <ul>
            <li><strong>Emerging Therapeutic Areas:</strong> Significant growth in oncology and rare disease research</li>
            <li><strong>Geographic Expansion:</strong> Opportunities in underserved regions for patient access</li>
            <li><strong>Technology Integration:</strong> Digital health and AI-driven trials showing promise</li>
            <li><strong>Regulatory Harmonization:</strong> Streamlined approval processes emerging globally</li>
            <li><strong>Investigator Networks:</strong> ML-identified top performers available for collaboration</li>
            </ul>
            </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)       
    
    with tab2:
        html_content = dedent(f"""
        <div class="warning-box">
            <h4>⚠️ Key Challenges</h4>
            <ul>
                <li><strong>Enrollment Difficulties:</strong> Patient recruitment remains a critical bottleneck</li>
                <li><strong>Diversity Gaps:</strong> Need for improved representation across demographics</li>
                <li><strong>Cost Escalation:</strong> Rising trial costs impacting feasibility</li>
                <li><strong>Regulatory Complexity:</strong> Varying requirements across jurisdictions</li>
                <li><strong>Investigator Selection:</strong> Need for data-driven investigator identification</li>
            </ul>
        </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)
    
    with tab3:
        html_content = dedent(f"""
            <div class="insight-box">
                <h4>🚀 Strategic Recommendations</h4>
                <ul>
                    <li><strong>Invest in Patient-Centric Design:</strong> Improve trial accessibility and reduce burden</li>
                    <li><strong>Leverage Real-World Evidence:</strong> Integrate RWE to enhance trial efficiency</li>
                    <li><strong>Strengthen Global Partnerships:</strong> Collaborate for broader reach and shared resources</li>
                    <li><strong>Adopt Digital Solutions:</strong> Use technology for remote monitoring and data collection</li>
                    <li><strong>Focus on Diversity:</strong> Implement targeted strategies for inclusive enrollment</li>
                    <li><strong>Use ML for Investigator Selection:</strong> Leverage ranking algorithms to identify top performers</li>
                </ul>
            </div>
        """)
        st.markdown(html_content, unsafe_allow_html=True)
    
    # Trend analysis
    st.subheader("📊 Trend Analysis")
    
    # Create a comprehensive trend chart (simulated data for demonstration)
    years = list(range(2018, 2025))
    trends_data = {
        'Year': years,
        'Total Studies': [800, 850, 900, 920, 950, 980, 1000],
        'Completion Rate': [65, 67, 70, 72, 74, 75, 76],
        'Digital Trials': [5, 8, 12, 18, 25, 35, 45]
    }
    
    trends_df = pd.DataFrame(trends_data)
    
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Study Volume Growth', 'Completion Rate Improvement', 
                        'Digital Trial Adoption', 'Geographic Expansion'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Study volume
    fig_trends.add_trace(
        go.Scatter(x=trends_df['Year'], y=trends_df['Total Studies'], 
                    mode='lines+markers', name='Total Studies'),
        row=1, col=1
    )
    
    # Completion rate
    fig_trends.add_trace(
        go.Scatter(x=trends_df['Year'], y=trends_df['Completion Rate'], 
                    mode='lines+markers', name='Completion Rate (%)', 
                    line=dict(color='green')),
        row=1, col=2
    )
    
    # Digital trials
    fig_trends.add_trace(
        go.Bar(x=trends_df['Year'], y=trends_df['Digital Trials'], 
                name='Digital Trials (%)', marker_color='purple'),
        row=2, col=1
    )
    
    # Geographic data (placeholder)
    countries_growth = [15, 18, 22, 25, 28, 30, 32]
    fig_trends.add_trace(
        go.Scatter(x=trends_df['Year'], y=countries_growth, 
                    mode='lines+markers', name='Countries Participating',
                    line=dict(color='orange')),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=600, showlegend=False, 
                            title_text="Clinical Trial Ecosystem Trends")
    st.plotly_chart(fig_trends, use_container_width=True)



render_executive_summary_page()